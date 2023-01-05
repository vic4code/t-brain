from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from pathlib import Path
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class ExtraSumDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.data = {
                        'ids':[],
                        'inputs_q':[],
                        'inputs_r':[],
                        'indicators':[],
                        'targets_q':[],
                        'targets_r':[]
        }
        self.data_dir = data_dir
        self.transform = transform

        with open(Path(self.data_dir, "data.csv"), "r") as file:

            lines = file.readlines()
            for line in tqdm(lines[1:10]):                
                splitted = line.strip("\n").split(",")

                for word, key in zip(splitted, self.data.keys()):
                    word = " ".join(word_tokenize(word.replace('"""',''))).lower()
                    self.data[key].append(word)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return list(self.data.items())[idx]


class DataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        tokenizer_name = "",
        model_name_or_path = "bert-base-uncased",
        tokenizer_no_use_fast = False,
        gradient_checkpointing = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir

        # data transformations
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name
            if self.hparams.tokenizer_name
            else self.hparams.model_name_or_path,
            use_fast=(not self.hparams.tokenizer_no_use_fast),
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    @classmethod
    def get_input_ids(
        cls,
        tokenizer,
        src_txt,
        bert_compatible_cls=True,
        sep_token=None,
        cls_token=None,
        max_length=None,
    ):
        """
        Get ``input_ids`` from ``src_txt`` using ``tokenizer``. See
        :meth:`~data.SentencesProcessor.get_features` for more info.
        """
        sep_token = str(sep_token)
        cls_token = str(cls_token)
        if max_length is None:
            try:
                max_length = list(tokenizer.max_model_input_sizes.values())[0]
            except AttributeError:
                max_length = tokenizer.model_max_length

        if max_length > 1_000_000:
            logger.warning(
                "Tokenizer maximum length is greater than 1,000,000. This is likely a mistake. "
                + "Resetting to 512 tokens."
            )
            max_length = 512

        # adds a '[CLS]' token between each sentence and outputs `input_ids`
        if bert_compatible_cls:
            # If the CLS or SEP tokens exist in the document as part of the dataset, then
            # set them to UNK
            unk_token = str(tokenizer.unk_token)
            src_txt = [
                sent.replace(sep_token, unk_token).replace(cls_token, unk_token)
                for sent in src_txt
            ]

            if not len(src_txt) < 2:  # if there is NOT 1 sentence
                # separate each sentence with ' [SEP] [CLS] ' (or model equivalent tokens) and
                # convert to string
                separation_string = " " + sep_token + " " + cls_token + " "
                text = separation_string.join(src_txt)
            else:
                try:
                    text = src_txt[0]
                except IndexError:
                    text = src_txt

            # tokenize
            src_subtokens = tokenizer.tokenize(text)
            # select first `(max_length-2)` tokens (so the following line of tokens can be added)
            src_subtokens = src_subtokens[: (max_length - 2)]
            # Insert '[CLS]' at beginning and append '[SEP]' to end (or model equivalent tokens)
            src_subtokens.insert(0, cls_token)
            src_subtokens.append(sep_token)
            # create `input_ids`
            input_ids = tokenizer.convert_tokens_to_ids(src_subtokens)
        else:
            input_ids = tokenizer.encode(
                src_txt,
                add_special_tokens=True,
                max_length=min(max_length, tokenizer.max_len),
            )

        return input_ids

    def get_features_process(
        self,
        input_information,
        num_examples=0,
        tokenizer=None,
        bert_compatible_cls=False,
        sep_token=None,
        cls_token=None,
        create_sent_rep_token_ids=True,
        sent_rep_token_id=None,
        create_sent_lengths=True,
        create_segment_ids="binary",
        segment_token_id=None,
        create_source=False,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        create_attention_mask=True,
        pad_ids_and_attention=True,
    ):
        """
        The process that actually creates the features.
        :meth:`~data.SentencesProcessor.get_features` is the driving function, look there for a
        description of how this function works. This function only exists so that processing can
        easily be done in parallel using ``Pool.map``.
        """
  
        if bert_compatible_cls:
            # convert `example.text` to array of sentences
            src_txt_q = (" ".join(sent) for sent in input_information['inputs_q'])
            src_txt_r = (" ".join(sent) for sent in input_information['inputs_r'])
        else:
            src_txt_q = input_information['inputs_q']
            src_txt_r = input_information['inputs_r']

        input_ids_q = self.get_input_ids(
            tokenizer, src_txt_q, bert_compatible_cls, sep_token, cls_token, max_length
        )

        input_ids_r = self.get_input_ids(
            tokenizer, src_txt_r, bert_compatible_cls, sep_token, cls_token, max_length
        )

        # Segment (Token Type) IDs
        segment_ids_q = None
        segment_ids_r = None

        if create_segment_ids == "binary":
            current_segment_flag = True
            segment_ids_q, segment_ids_r = [], []
            for segment_ids, token in zip([segment_ids_q, segment_ids_r], [input_ids_q, input_ids_r]):
                segment_ids += [0 if current_segment_flag else 1]
                if token == segment_token_id:
                    current_segment_flag = not current_segment_flag

        if create_segment_ids == "sequential":
            current_segment = 0
            segment_ids_q, segment_ids_r = [], []
            for segment_ids, token in zip([segment_ids_q, segment_ids_r], [input_ids_q, input_ids_r]):
                segment_ids += [current_segment]
                if token == segment_token_id:
                    current_segment += 1

        # Sentence Representation Token IDs and Sentence Lengths
        sent_rep_ids_q = None
        sent_rep_ids_r = None

        sent_lengths_q = None
        sent_lengths_r = None

        if create_sent_rep_token_ids:
            # create list of indexes for the `sent_rep` tokens
            sent_rep_ids = [
                i for i, t in enumerate(input_ids) if t == sent_rep_token_id
            ]
            # truncate `label` to the length of the `sent_rep_ids` aka the number of sentences
            label = label[: len(sent_rep_ids)]

            if create_sent_lengths:
                # if there are 1 or 0 sentences then the length of the entire sequence will be
                # the only value in `sent_lengths`
                if len(sent_rep_ids) < 2:
                    sent_lengths = [len(input_ids)]
                else:
                    sent_lengths = [
                        sent_rep_ids[i] - sent_rep_ids[i - 1]
                        for i in range(1, len(sent_rep_ids))
                    ]
                    # Add sentence length for the last sentence, if missing.
                    # If the last sentence representation token position in `input_ids` is not
                    # the last token in `input_ids` then add the length of the last sentence
                    # to `sent_lengths` by subtracting the position of the last `sent_rep_token`
                    # from the length of `input_ids`
                    if sent_rep_ids[-1] != len(input_ids) - 1:
                        sent_lengths.append(len(input_ids) - sent_rep_ids[-1])
                    # Add sentence length for the first sentence, if missing.
                    # If the first sentence representation token is not the first token in
                    # `input_ids` then add the length of the first sentence by inserting
                    # the first value in `sent_rep_ids` at the front of `sent_lengths`.
                    if sent_rep_ids[0] != 0:
                        sent_lengths.insert(0, sent_rep_ids[0] + 1)

        # Attention
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        if create_attention_mask:
            attention_mask_q = [1 if mask_padding_with_zero else 0] * len(input_ids_q)
            attention_mask_r = [1 if mask_padding_with_zero else 0] * len(input_ids_r)

        # Padding
        # Zero-pad up to the sequence length.
        if pad_ids_and_attention:
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )

            assert (
                len(input_ids) == max_length
            ), "Error with input length {} vs {}".format(len(input_ids), max_length)
            assert (
                len(attention_mask) == max_length
            ), "Error with input length {} vs {}".format(
                len(attention_mask), max_length
            )

        # Return features
        # if the attention mask was created then add the mask to the returned features
        outputs = {
            "input_ids": input_ids,
            "labels": label,
            "token_type_ids": segment_ids,
            "sent_rep_token_ids": sent_rep_ids,
            "sent_lengths": sent_lengths,
            "target": example.target,
        }
        if create_attention_mask:
            outputs["attention_mask"] = attention_mask
        if create_source:
            # convert form individual tokens to only individual sentences
            source = [" ".join(sentence) for sentence in example.text]
            outputs["source"] = source

        return outputs


    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        
        dataset = ExtraSumDataset(self.hparams.data_dir)
        self.features = []

        for input_information in dataset.data.items():
            self.features.append(self.get_features_process(input_information=input_information, tokenizer=self.tokenizer))


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = ExtraSumDataset(self.hparams.data_dir)
            print(dataset.data)
            return
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "data.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
    
    dataset = DataModule(train_val_test_split=cfg.train_val_test_split)
    dataset.setup()
    # for batch_ndx, sample in enumerate(dataset.data_train):
    #     print(sample)
    #     break

    print(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_name
        if cfg.tokenizer_name
        else cfg.model_name_or_path,
        use_fast=(not cfg.tokenizer_no_use_fast),
    )

    print(tokenizer)
    input_strings = ["hello world", "how old are you", "I'm fine fuck you"]
    output = dataset.get_input_ids(tokenizer, input_strings)
    print(output)

    print(dataset.data_train)