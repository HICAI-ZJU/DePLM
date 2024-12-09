from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import logging
import torch
import esm
import esm.inverse_folding
import ast
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader


LOG = logging.getLogger(__name__)


class BatchedProteinGymSubstitutionDataset(Dataset):
    def __init__(self, assay_name, wt_sequence, coords, train_data, valid_data):
        super().__init__()
        self.assay_name = assay_name
        _, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        _, self.structure_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        self.coord_converter = esm.inverse_folding.util.CoordBatchConverter(self.structure_alphabet)
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
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[:1]], dtype=torch.long), torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                train_label.append((torch.tensor(data['DMS_score'], dtype=torch.float32), mutant_list))
            self.train_labels.append(train_label)
        
        self.valid_labels = []
        for assay_idx, dms_data in enumerate(valid_data):
            valid_label = []
            for index, data in dms_data.iterrows():
                mutants = data['mutant'].split(':')
                mutant_list = []
                for mutant in mutants:
                    location = int(mutant[1:-1])
                    mutant_list.append((location, torch.tensor(self.alphabet.tok_to_idx[mutant[:1]], dtype=torch.long), torch.tensor(self.alphabet.tok_to_idx[mutant[-1:]], dtype=torch.long)))
                valid_label.append((torch.tensor(data['DMS_score'], dtype=torch.float32), mutant_list))
            self.valid_labels.append(valid_label)

    def __getitem__(self, index):
        return self.assay_name, self.batch_tokens, self.coords, self.train_labels, self.valid_labels
    

    def __len__(self):
        return 1


class BatchedProteinGymSubstitutionDataModule(LightningDataModule):
    """`LightningDataModule` for the ProteinGym dataset.
    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        assay_index: int = 0, # 0 - 100
        split_type: str = "random", # random, modulo, contiguous
        split_index: int = 0, # 0 - 4
        support_assay_num: int = 40,
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
            self.hparams.data_dir = Path(self.hparams.data_dir) / 'ProteinGym'
            assay_reference_file = pd.read_csv(self.hparams.data_dir/'reference'/'DMS_substitutions.csv')
            assay_id = assay_reference_file["DMS_id"][self.hparams.assay_index]
            assay_file_name = assay_reference_file["DMS_filename"][assay_reference_file["DMS_id"]==assay_id].values[0]
            pdb_file_name = assay_reference_file["pdb_file"][assay_reference_file["DMS_id"]==assay_id].values[0]
            assay_data = pd.read_csv(self.hparams.data_dir/'substitutions'/assay_file_name)
            wt_sequence = assay_reference_file["target_seq"][assay_reference_file["DMS_id"]==assay_id].values[0]
            structure = esm.inverse_folding.util.load_structure(str(self.hparams.data_dir/'structure'/pdb_file_name), 'A')
            coord, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)

            ### For generalization test
            # skip_nums=[0, 24, 25, 29, 30, 31, 58, 86, 103, 104, 128, 130, 175, 184, 185, 207]
            # with open(Path(self.hparams.data_dir) / 'ProteinGym' / 'cluster/id_0.5_cov_0.8.txt', 'r') as f_in:
            #     cluster_set = ast.literal_eval(f_in.readline())
            # test_assay_nums = None
            # for cluster in cluster_set:
            #     if self.hparams.assay_index in cluster:
            #         test_assay_nums = cluster
            # assay_selection_type = assay_reference_file["coarse_selection_type"][assay_reference_file["DMS_id"]==assay_id].values[0]
            # support_assay_ids = assay_reference_file["DMS_id"][assay_reference_file['coarse_selection_type']==assay_selection_type]
            # support_assay_ids = support_assay_ids[support_assay_ids.index != self.hparams.assay_index]
            # for skip_num in skip_nums:
            #     support_assay_ids = support_assay_ids[support_assay_ids.index != skip_num]
            # for test_assay_id in test_assay_nums:
            #     support_assay_ids = support_assay_ids[support_assay_ids.index != test_assay_id]
            # support_assay_file_names = [assay_reference_file["DMS_filename"][assay_reference_file["DMS_id"]==support_assay_id].values[0] for support_assay_id in support_assay_ids]
            # support_pdb_file_names = [assay_reference_file["pdb_file"][assay_reference_file["DMS_id"]==support_assay_id].values[0] for support_assay_id in support_assay_ids]
            # support_assay_data = [pd.read_csv(self.hparams.data_dir/'substitutions'/support_assay_file_name) for support_assay_file_name in support_assay_file_names]
            # support_wt_sequences = [assay_reference_file["target_seq"][assay_reference_file["DMS_id"]==support_assay_id].values[0] for support_assay_id in support_assay_ids]
            # support_structures = [esm.inverse_folding.util.load_structure(str(self.hparams.data_dir/'structure'/support_pdb_file_name), 'A') for support_pdb_file_name in support_pdb_file_names]
            # support_coords = [esm.inverse_folding.util.extract_coords_from_structure(support_structure)[0] for support_structure in support_structures]
            # support_assay_indices = list(support_assay_ids.index)
            # if len(support_assay_indices) > self.hparams.support_assay_num:
            #     support_assay_indices, support_wt_sequences, support_coords, support_assay_data = support_assay_indices[:self.hparams.support_assay_num], support_wt_sequences[:self.hparams.support_assay_num], support_coords[:self.hparams.support_assay_num], support_assay_data[:self.hparams.support_assay_num]

            # self.data_train = BatchedProteinGymSubstitutionDataset(support_assay_indices, support_wt_sequences, support_coords,
            #      support_assay_data, 
            #      [])
            # self.data_val = BatchedProteinGymSubstitutionDataset([assay_id], [wt_sequence], [coord],
            #      [], 
            #      [assay_data])

            self.data_train = BatchedProteinGymSubstitutionDataset([assay_id], [wt_sequence], [coord],
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] != self.hparams.split_index].reset_index()], 
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] == self.hparams.split_index].reset_index()])
            self.data_val = BatchedProteinGymSubstitutionDataset([assay_id], [wt_sequence], [coord],
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] != self.hparams.split_index].reset_index()], 
                 [assay_data[assay_data[f"fold_{self.hparams.split_type}_5"] == self.hparams.split_index].reset_index()])
            LOG.info(f'Target assay {assay_id}; Length: {len(wt_sequence)}')

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
