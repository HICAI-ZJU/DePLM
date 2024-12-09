import copy
import logging
import numpy as np
from scipy import stats
from typing import Any, Dict, Tuple
import random
import torch
import esm
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric, PearsonCorrCoef
import torchsort

def spearmanr(pred, target, **kw):
    pred = torchsort.soft_rank(pred, **kw)
    target = torchsort.soft_rank(target, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

LOG = logging.getLogger(__name__)

from src.models.DePLM_components.modules import TriangularSelfAttentionBlock
from src.models.DePLM_components.sort import quick_sort, _rank_data

from torch import nn
from torch.nn import LayerNorm


class DePLM4ProteinEngineeringModule(LightningModule):
    def __init__(
        self,
        diff_total_step,
        model,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        if model == 'esm2':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif model == 'esm1v':
            self.model, self.alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
            self.model.embed_dim = self.model.args.embed_dim
            self.model.attention_heads = self.model.args.attention_heads

        self.structure_model, self.structure_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.model, self.structure_model = self.model.eval(), self.structure_model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        for name, param in self.structure_model.named_parameters():
            param.requires_grad = False

        self.state = {}
        
        self.val_spearman = MeanMetric()
        self.val_spearman_best = MaxMetric()
        self.criterion = PearsonCorrCoef()

        self.diff_total_step = diff_total_step
        self.automatic_optimization = False

        self.repr_dim = self.model.embed_dim
        self.repr_combine = nn.Parameter(torch.zeros(self.model.num_layers + 1))
        self.repr_mlp = nn.Sequential(LayerNorm(self.repr_dim), nn.Linear(self.repr_dim, self.repr_dim), nn.GELU(), nn.Linear(self.repr_dim, self.repr_dim))
        self.structure_repr_mlp = nn.Sequential(LayerNorm(self.structure_model.encoder.args.encoder_embed_dim), nn.Linear(self.structure_model.encoder.args.encoder_embed_dim, self.repr_dim), nn.GELU(), nn.Linear(self.repr_dim, self.repr_dim))
        self.attn_dim = 32
        self.attn_num = self.model.num_layers * self.model.attention_heads
        self.attn_mlp = nn.Sequential(LayerNorm(self.attn_num), nn.Linear(self.attn_num, self.attn_num), nn.GELU(), nn.Linear(self.attn_num, self.attn_dim))

        self.num_blocks = 1
        self.blocks = nn.ModuleList(
            [
                TriangularSelfAttentionBlock(
                    sequence_state_dim=self.repr_dim,
                    pairwise_state_dim=self.attn_dim,
                    sequence_head_width=32,
                    pairwise_head_width=32,
                    dropout=0.2
                ) for _ in range(self.num_blocks)
            ]
        )
        self.step_embedding = nn.Embedding(self.diff_total_step, self.repr_dim,)
        self.logits_mlp = nn.Sequential(nn.Linear(self.model.alphabet_size, self.repr_dim),  nn.GELU(), LayerNorm(self.repr_dim))
        self.conv = nn.Conv1d(self.repr_dim, self.repr_dim, kernel_size=7, stride=1, padding=3)
        self.logits_representation_mlp = nn.Sequential(nn.Linear(2 * self.repr_dim, self.repr_dim))
        
    def state_setup(self, x_wt, y_label):
        y = []
        locations = set()
        for score, mutants in y_label:
            y.append({
                'score': score,
                'mutants': [(mutant[0], mutant[1], mutant[2]) for mutant in mutants]
            })
            for mutant in mutants:
                locations.add(mutant[0])

        masked_batch_tokens = x_wt.clone()
        with torch.no_grad():
            result = self.model(masked_batch_tokens, repr_layers=range(self.model.num_layers+1), need_head_weights=True)
        x = {'input': x_wt, 'logits': result['logits'][0], 'representation': torch.stack([v for _, v in sorted(result['representations'].items())], dim=2), 'attention': result['attentions'].permute(0, 4, 3, 1, 2).flatten(3, 4)}
        return (x, y)

    def forward(self, x, structure_repr):
        return_logits = []

        logits, representation, attention = x['logits'], x['representation'], x['attention']
        residx = torch.arange(x['input'].shape[1], device=self.device).expand_as(x['input'])
        mask = torch.ones_like(x['input'])

        representation = self.repr_mlp((self.repr_combine.softmax(0).unsqueeze(0) @ representation).squeeze(2)) + self.structure_repr_mlp(structure_repr).repeat(representation.shape[0], 1, 1)
        attention = self.attn_mlp(attention)

        def trunk_iter(s, z, residx, mask):
            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx, chunk_size=None)
            return s, z
        
        representation, attention = trunk_iter(representation, attention, residx, mask)
        logits_recycle = logits
        return_logits.append(logits_recycle)
        step_embed = self.step_embedding(torch.tensor([i for i in range(self.diff_total_step)], dtype=torch.long).to(representation.device))
        for diff_step in range(self.diff_total_step):
            logits_representation = self.logits_mlp(logits_recycle.detach()) + step_embed[diff_step]
            logits_representation = self.conv(logits_representation.transpose(0,1)).transpose(0,1)
            logits_recycle = logits_recycle - self.model.lm_head(self.logits_representation_mlp(torch.cat([logits_representation, representation[0]], dim=-1)))
            return_logits.append(logits_recycle)

        return return_logits, representation

    def loss_compute_and_backward(self, x, y, structure_repr):
        opt = self.optimizers()
        x_logits, _ = self.forward(x, structure_repr=structure_repr)
        combined_x_logits = [torch.stack([sum([x_logits[step][mutant[0], mutant[2]] - x_logits[step][mutant[0], mutant[1]] for mutant in y[index]['mutants']]) / len(y[index]['mutants']) for index in range(len(y))], dim=-1) for step in range(self.diff_total_step + 1)]
        combined_y_scores = torch.stack([y[index]['score'] for index in range(len(y))])
        loss = 0.
        for index in range(1, self.diff_total_step+1):
            combined_intermediate_ranked_y_scores = self.intermediate_score_compute(combined_x_logits[index-1], combined_y_scores)
            if index == self.diff_total_step:
                combined_intermediate_ranked_y_scores = combined_intermediate_ranked_y_scores[-1]
            else:
                combined_intermediate_ranked_y_scores = combined_intermediate_ranked_y_scores[index-1]

            loss += (1 - spearmanr(combined_x_logits[index].unsqueeze(0), combined_intermediate_ranked_y_scores.unsqueeze(0)))
        self.manual_backward(loss)

        opt.step()
        opt.zero_grad()
        spearman = stats.spearmanr(combined_x_logits[-1].detach().cpu(), combined_y_scores.detach().cpu()).statistic
        return loss, spearman
    
    def output_process(self, x_logits, y):
        x_logits = torch.stack([sum([x_logits[mutant[0], mutant[2]] - x_logits[mutant[0], mutant[1]] for mutant in y[index]['mutants']]) / len(y[index]['mutants']) for index in range(len(y))], dim=-1)
        y_scores = torch.stack([y[index]['score'] for index in range(len(y))])
        return x_logits, y_scores
    
    def on_train_start(self) -> None:
        self.val_spearman.reset()
        self.val_spearman_best.reset()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        assay_names, batch_tokens, coords, train_labels, _ = batch

        for name, x_wt, coord, train_label in zip(assay_names, batch_tokens, coords, train_labels):
            if f'{name}-train' not in self.state:
                self.state[f'{name}-train'] = self.state_setup(x_wt, train_label)
            if f'{name}-structure' not in self.state:
                self.state[f'{name}-structure'] = self.structure_model.encoder.forward(*coord)['encoder_out'][0].transpose(0, 1)
            x, y = self.state[f'{name}-train']
            loss, spearman = self.loss_compute_and_backward(x, y, structure_repr=self.state[f'{name}-structure'])
            LOG.info(f'Training assay {name}: loss {loss}; spearman {spearman}')

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        self.val_spearman.reset()
        assay_names, batch_tokens, coords, _, valid_labels = batch

        for name, x_wt, coord, valid_label in zip(assay_names, batch_tokens, coords, valid_labels):
            if f'{name}-valid' not in self.state:
                self.state[f'{name}-valid'] = self.state_setup(x_wt, valid_label)
            if f'{name}-structure' not in self.state:
                self.state[f'{name}-structure'] = self.structure_model.encoder.forward(*coord)['encoder_out'][0].transpose(0, 1)
            x, y = self.state[f'{name}-valid']
            x_logits, _ = self.forward(x, structure_repr=self.state[f'{name}-structure'])
            x_logits = x_logits[-1]
            x_logits, y_scores = self.output_process(x_logits, y)
            spearman = stats.spearmanr(x_logits.detach().cpu(), y_scores.detach().cpu()).statistic
            LOG.info(f'Testing assay {name}: spearman {spearman}')
            self.val_spearman(spearman)
    
    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        self.log("val/spearman", self.val_spearman.compute(), sync_dist=True, prog_bar=True)

        spearman = self.val_spearman.compute()  # get current val acc
        self.val_spearman_best(spearman)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/spearman_best", self.val_spearman_best.compute(), sync_dist=True, prog_bar=True)

    def intermediate_score_compute(self, x_logits, y_scores):
        ranked_x_logits = (_rank_data(x_logits) - 1)
        ranked_y_scores = (_rank_data(y_scores) - 1)
        sorted_index = torch.argsort(ranked_y_scores)
        begin_state = ranked_x_logits[sorted_index]
        end_state = ranked_y_scores[sorted_index]
        intermediate_states = torch.tensor(quick_sort(begin_state.tolist()) + [end_state.tolist()], dtype=x_logits.dtype, device=self.device)
        intermediate_ranked_y_scores = []
        for state in intermediate_states:
            ranked_y_score = torch.empty([state.shape[0]], dtype=x_logits.dtype, device=self.device)
            ranked_y_score[sorted_index] = state
            intermediate_ranked_y_scores.append(ranked_y_score)
        return intermediate_ranked_y_scores

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/spearman_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
