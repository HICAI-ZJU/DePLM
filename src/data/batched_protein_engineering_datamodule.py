from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import logging
import torch
import esm
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

LOG = logging.getLogger(__name__)


class BatchedProteinEngineeringDataset(Dataset):
    def __init__(self, name, wt_sequence, coords, train_data, valid_data):
        _, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        _, self.structure_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.coord_converter = esm.inverse_folding.util.CoordBatchConverter(self.structure_alphabet)
        self.name = name
        self.coords = []
        for coord in coords:
            batch = [(coord, None, None)]
            coord, confidence, _, _, padding_mask = self.coord_converter(batch)
            self.coords.append((coord, padding_mask, confidence))
        
        self.batch_tokens = []
        for sequence in wt_sequence:
            batch_labels, batch_strs, batch_tokens = self.batch_converter([('protein', sequence)])
            self.batch_tokens.append(batch_tokens)

        self.train_labels = []
        for assay_idx, dms_data in enumerate(train_data):
            train_label = []
            for index, data in dms_data.iterrows():
                mutants = data['mutant'].split(':')
                mutant_list = []
                for mutant in mutants:
                    location = int(mutant[1:-1])
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                train_label.append((torch.tensor(data['score'], dtype=torch.float32), mutant_list))
            self.train_labels.append(train_label)

        self.valid_labels = []
        for assay_idx, dms_data in enumerate(valid_data):
            valid_label = []
            for index, data in dms_data.iterrows():
                mutants = data['mutant'].split(':')
                mutant_list = []
                for mutant in mutants:
                    location = int(mutant[1:-1])
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                valid_label.append((torch.tensor(data['score'], dtype=torch.float32), mutant_list))
            self.valid_labels.append(valid_label)

    def __getitem__(self, index):
        return self.name, self.batch_tokens, self.coords, self.train_labels, self.valid_labels    

    def __len__(self):
        return 1



class BatchedProteinEngineeringDataModule(LightningDataModule):
    """`LightningDataModule` for the ProteinGym dataset.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        task_name: str,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        _, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 1

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        """

        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.hparams.data_dir = Path(self.hparams.data_dir) / self.hparams.task_name
            with open(self.hparams.data_dir/'wildtype.txt', 'r') as f:
                wt_sequence = f.readline().strip()
            assay_data = pd.read_csv(self.hparams.data_dir/f'{self.hparams.task_name}.csv')
            structure = esm.inverse_folding.util.load_structure(str(self.hparams.data_dir/f'{self.hparams.task_name}.pdb'), 'A')
            coord, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)

            self.data_train = BatchedProteinEngineeringDataset([self.hparams.task_name], [wt_sequence], [coord],
                 [assay_data[assay_data["split"] == 0].reset_index()], 
                 [assay_data[assay_data["split"] == 2].reset_index()])
            self.data_val = BatchedProteinEngineeringDataset([self.hparams.task_name], [wt_sequence], [coord],
                 [assay_data[assay_data["split"] == 0].reset_index()], 
                 [assay_data[assay_data["split"] == 2].reset_index()])

    def collator(self, raw_batch):
        return raw_batch[0]

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=False
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
