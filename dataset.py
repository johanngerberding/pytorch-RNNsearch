import torch
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k, IWSLT2016, IWSLT2017
from torchtext.vocab import build_vocab_from_iterator  
from torch.nn.utils.rnn import pad_sequence

from typing import Iterable, List, Tuple


def yield_tokens(data_iter: Iterable, language: str, token_transform: dict, cfg):
    language_index = {cfg.DATASET.SRC_LANGUAGE: 0, cfg.DATASET.TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


def tensor_transform(token_ids: List[int], cfg):
    "Add BOS/EOS and create tensor for input sequence indices"
    return torch.cat((torch.tensor([cfg.DATASET.BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([cfg.DATASET.EOS_IDX])))


def prepare_data(cfg, special_first: bool = True) -> Tuple[dict, dict, dict]:
    token_transform = {}
    vocab_transform = {}
    text_transform = {}

    # pip install -U spacy
    # python -m spacy download en_core_web_sm
    # python -m spacy download de_core_news_sm
    token_transform[cfg.DATASET.SRC_LANGUAGE] = get_tokenizer("spacy", language="de_core_news_sm")
    token_transform[cfg.DATASET.TGT_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")

    for ln in [cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE]:
        
        if cfg.DATASET.NAME == "Multi30k":
            train_iter = Multi30k(
                split="train", 
                language_pair=(cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE)
            )
        elif cfg.DATASET.NAME == "IWSLT2016":
            train_iter = IWSLT2016(
                split="train", 
                language_pair=(cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE)
            ) 
        elif cfg.DATASET.NAME == "IWSLT2017":
            train_iter = IWSLT2017(
                split="train", 
                language_pair=(cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE)
            )
        else:
            print("Please use one of the following three Pytorch datasets:")
            print("-> Multi30k")
            print("-> IWSLT2016")
            print("-> IWSLT2017")
            print("Alternatively create your own dataset.py")
            raise Exception

        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln, token_transform, cfg), 
                                                        min_freq=cfg.DATASET.MIN_FREQ, 
                                                        specials=cfg.DATASET.SPECIAL_SYMBOLS, 
                                                        special_first=special_first)

        print("Length of vocab [{} - {}]: {}".format(
            ln, cfg.DATASET.NAME, len(vocab_transform[ln])))
    
    # set UNK as default index
    for ln in [cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(cfg.DATASET.UNK_IDX)

    t_transform = lambda tokens : tensor_transform(tokens, cfg)

    for ln in [cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(
            token_transform[ln], # Tokenization 
            vocab_transform[ln], # Numericalization
            t_transform, # add BOS/EOS and create tensor
        )
    
    return token_transform, vocab_transform, text_transform


class Collater:
    def __init__(self, cfg, text_transform):
        self.cfg = cfg 
        self.text_transform = text_transform
    
    def __call__(self, batch):
        "Collate data samples into batch tensors"
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch: 
            src_batch.append(self.text_transform[self.cfg.DATASET.SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[self.cfg.DATASET.TGT_LANGUAGE](tgt_sample.rstrip("\n")))
        src_batch = pad_sequence(src_batch, padding_value=self.cfg.DATASET.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.cfg.DATASET.PAD_IDX)

        return src_batch, tgt_batch 


def get_iter(cfg, split: str = "train"):
    if cfg.DATASET.NAME == "Multi30k":
        from torchtext.datasets import Multi30k  
        iter = Multi30k(
            split=split, 
            language_pair=(cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE)
        )
    elif cfg.DATASET.NAME == "IWSLT2016":
        from torchtext.datasets import IWSLT2016  
        iter = IWSLT2016(
            split=split, 
            language_pair=(cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE)
        ) 
    elif cfg.DATASET.NAME == "IWSLT2017":
        from torchtext.datasets import IWSLT2017  
        iter = IWSLT2017(
            split=split, 
            language_pair=(cfg.DATASET.SRC_LANGUAGE, cfg.DATASET.TGT_LANGUAGE)
        )
    else:
        print("Please specify one of the following three Pytorch datasets:")
        print("-> Multi30k")
        print("-> IWSLT2016")
        print("-> IWSLT2017")
        print("in your config.yaml.")
        print("Alternatively create your own dataset.py.")
        raise Exception
    
    return iter