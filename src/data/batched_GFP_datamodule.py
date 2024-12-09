from pathlib import Path
from typing import Any, Dict, Optional, List
import esm.inverse_folding
import pandas as pd
import logging
import torch
import esm
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


LOG = logging.getLogger(__name__)


def mutant_process(mutants, start_mutants=None):
    if start_mutants is None:
        return mutants
    return_mutants = []
    start_mutant_dict = {int(m[1:-1]): m for m in start_mutants}
    for mutant in mutants:
        location = int(mutant[1:-1])
        if location not in start_mutant_dict.keys():
            return_mutants.append(mutant)
            continue
        wt = start_mutant_dict[location][0]
        mt = mutant[-1]
        if wt == mt:
            continue
        return_mutants.append(f'{wt}{location}{mt}')
        del start_mutant_dict[location]
    return_mutants = return_mutants + list(start_mutant_dict.values())
    return return_mutants

class BatchedGFPDataset(Dataset):
    def __init__(self, name, wt_sequence, coords, train_data, val_data):
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
                mutants = mutant_process(mutants, ['S38T', 'R41K', 'S105N'])
                mutant_list = []
                for mutant in mutants:
                    location = int(mutant[1:-1])
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[:1]], dtype=torch.long), torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                train_label.append((torch.tensor(data['score'], dtype=torch.float32), mutant_list))
            self.train_labels.append(train_label)

        self.val_labels = []
        for assay_idx, dms_data in enumerate(val_data):
            val_label = []
            for index, data in dms_data.iterrows():
                mutants = data['mutant'].split(':')
                if len(mutants) > 3:
                    continue
                mutant_list = []
                for mutant in mutants:
                    location = int(mutant[1:-1])
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[:1]], dtype=torch.long), torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                val_label.append((torch.tensor(data['score'], dtype=torch.float32), mutant_list))
            self.val_labels.append(val_label)
        LOG.info(f'train data: {len(self.train_labels)}; val data: {len(self.val_labels)}')

    def __getitem__(self, index):
        return self.name, self.batch_tokens, self.coords, self.train_labels, self.val_labels

    def __len__(self):
        return 1


class BatchedGFPDataModule(LightningDataModule):
    """`LightningDataModule` for the GFP dataset.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        task_name: str,
        support_name: List,
        data_dir: str = "data/",
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        _, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()

    def setup(self, stage: Optional[str] = None) -> None:
        if not self.data_train:
            self.hparams.data_dir = Path(self.hparams.data_dir) / 'GFP'
            with open(self.hparams.data_dir / f'{self.hparams.task_name}.txt', 'r') as f_in:
                wt_sequence = f_in.readline().strip()
            wt_assay_data = pd.read_csv(self.hparams.data_dir/f'{self.hparams.task_name}.csv')
            wt_structure = esm.inverse_folding.util.load_structure(str(self.hparams.data_dir/f'{self.hparams.task_name}.pdb'), 'A')
            wt_coord, _ = esm.inverse_folding.util.extract_coords_from_structure(wt_structure)

            support_name = []
            support_wt_sequence = []
            support_assay_data = []
            support_coord = []
            for name in self.hparams.support_name:
                support_name.append(name)
                with open(self.hparams.data_dir / f'{name}.txt', 'r') as f_in:
                    support_wt_sequence.append(f_in.readline().strip())
                support_assay_data.append(pd.read_csv(self.hparams.data_dir/f'{name}_mt.csv'))
                structure = esm.inverse_folding.util.load_structure(str(self.hparams.data_dir/f'{name}_mt.pdb'), 'A')
                coord, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
                support_coord.append(coord)

            self.data_train = BatchedGFPDataset([self.hparams.task_name] + support_name, [wt_sequence] + support_wt_sequence, [wt_coord] + support_coord,
                 [wt_assay_data[wt_assay_data["split"] == 0].reset_index()] + support_assay_data, 
                 [wt_assay_data[wt_assay_data["split"] == 2].reset_index()])

            self.data_val = BatchedGFPDataset([self.hparams.task_name], [wt_sequence], [wt_coord],
                 [wt_assay_data[wt_assay_data["split"] == 0].reset_index()], 
                 [wt_assay_data[wt_assay_data["split"] == 2].reset_index()])

    def collator(self, raw_batch):
        return raw_batch[0]

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collator,
            shuffle=True
        )
    
    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
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