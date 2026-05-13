from datasets import load_dataset
import os
import re
from transformers import GPT2TokenizerFast, AutoTokenizer
from datasets import load_dataset
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
from datasets import Dataset

from torch.utils.data import DataLoader, DistributedSampler


def cycle_loader(dataloader, sampler=None):
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()

def get_dataset(name, mode, cache_dir=None, block_size=1024, num_proc=len(os.sched_getaffinity(0)), take=-1, varylen=True):
    if name == "wikitext103":
        dataset = load_dataset("wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir)
    elif name == "wikitext2":
        dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir)
    elif name == "ptb":
        dataset = load_dataset("ptb_text_only", cache_dir=cache_dir)
    elif name == "lambada":
        dataset = get_lambada_test_dataset()
    elif name == 'openwebtext': # 
        dataset = load_dataset('openwebtext', cache_dir=cache_dir)
        if take != -1:
            dataset = {k:v.take(take) if k=='train' else v for (k,v) in dataset.items()}
    else:
        dataset = load_dataset(name, cache_dir=cache_dir)
        # dataset = {k:v.take(10000) if k=='train' else v for (k,v) in dataset.items()}

    if name == "lambada":
        data = dataset
    else:
        data = dataset[mode]

    if name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif name == "ptb":
        detokenizer = ptb_detokenizer
    elif name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif name == "lambada":
        detokenizer = lambada_detokenizer
    else:
        detokenizer = None

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                 text[i] = detokenizer(t)
            return text
        return detok

    # tokenizer = GPT2TokenizerFast.from_pretrained('data/gpt2')
    # EOS = tokenizer.encode(tokenizer.eos_token)[0]
    # BOS = tokenizer.encode(tokenizer.bos_token)[0] ### MDLM
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #GPT2TokenizerFast.from_pretrained('gpt2')
    BOS = tokenizer.cls_token_id
    EOS = tokenizer.sep_token_id

    def preprocess_and_tokenize(example):
        if name == "ptb":
            text = example['sentence']
        else:
            text = example["text"]
        
        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        tokens = tokenizer(
            text, return_attention_mask=False) # => cls + ex + sep 
        # add in EOS token following 
        # https://github.com/jcpeterson/openwebtext/blob/master/tokenize_text.py#L67
        # Safe to concat.
        for token in tokens['input_ids']: # remove the first and last tokens automatically added by the tokenizer
            token.pop(0)
            token.pop(-1)
        return tokens
    
    tokenized_dataset = data.map(
        preprocess_and_tokenize, 
        batched=True, 
        num_proc=num_proc, 
        load_from_cache_file=True,
        desc='Tokenizing'
    )
    if name == "ptb":
        tokenized_dataset = tokenized_dataset.remove_columns('sentence')
    else:
        tokenized_dataset = tokenized_dataset.remove_columns('text')
        tokenized_dataset = tokenized_dataset.remove_columns('token_type_ids')

    if name=='tiny_roc_stories':
        tokenized_dataset = tokenized_dataset.remove_columns('source')
         

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        new_block_size = block_size - 2  # [BOS] and [EOS] to be added
        total_length = (total_length // new_block_size) * new_block_size
        # Split by chunks of max_len.
        result = {
            k: [
                [BOS] + t[i : i + new_block_size] + [EOS]
                for i in range(0, total_length, new_block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        # import ipdb; ipdb.set_trace()
        return result

    def pad_texts(examples):
        def handle(ex):
            max_seq_len = block_size
            new_block_size = max_seq_len - 2
            if len(ex) < max_seq_len:
                return [[BOS] + ex + [EOS] * (max_seq_len - 1 - len(ex))]
            else:
                return [
                    [BOS] + ex[i : i + new_block_size] + [EOS]
                    for i in range(0, len(ex), new_block_size)
                ]# + [[BOS] + ex[len(ex) // new_block_size * new_block_size :]]

        result = {
            k: list(chain(*[
                handle(ex) for ex in t
            ]
            )) for k, t in examples.items()
        }
        return result

    def group_texts_no_special_token(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    def handle_long_texts(examples):
        # Maximum sequence length is 2048. 
        # When encountering sequences longer than that, truncate to the maximum length and then put the remaining part in the next sequence.
        # Do this until remaining length is smaller than maximum length.
        def handle(ex):
            max_seq_len = block_size
            new_block_size = max_seq_len - 2
            if len(ex) < max_seq_len:
                return [[BOS] + ex]
            else:
                # return []
                i=0
                return [
                    [BOS] + ex[i : i + new_block_size] 
                    # for i in range(0, len(ex), new_block_size)
                ]# + [[BOS] + ex[len(ex) // new_block_size * new_block_size :]]

        result = {
            k: list(chain(*[
                handle(ex) for ex in t
            ]
            )) for k, t in examples.items()
        }
        return result

    if varylen:
        chunked_dataset = tokenized_dataset.map(handle_long_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    else:
        if True:
            chunked_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
        else:
            chunked_dataset = tokenized_dataset.map(pad_texts, batched=True, num_proc=num_proc, load_from_cache_file=True)
    chunked_dataset = chunked_dataset.with_format('torch')
    return chunked_dataset


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")


    train_set = get_dataset(config.data.train, "train", cache_dir=config.data.cache_dir, block_size=config.model.length, take=config.data.take, varylen=config.training.varylen)
    valid_set = get_dataset(config.data.valid, "validation" if config.data.valid not in ["text8", "lm1b"] else "test", cache_dir=config.data.cache_dir, block_size=config.model.length, varylen=config.eval.varylen)

    if distributed:
        train_sampler = DistributedSampler(train_set) 
        test_sampler = DistributedSampler(valid_set)
    else:
        train_sampler = None
        test_sampler = None

    # # when raw dataset contains varying length, use collator for batching
    # tokenizer = GPT2TokenizerFast.from_pretrained('data/gpt2')
    # EOS = tokenizer.encode(tokenizer.eos_token)[0]
    # BOS = tokenizer.encode(tokenizer.bos_token)[0] ### MDLM
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') #GPT2TokenizerFast.from_pretrained('gpt2')
    BOS = tokenizer.cls_token_id
    EOS = tokenizer.sep_token_id
    from transformers import DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = cycle_loader(DataLoader(
        train_set,
        batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
        collate_fn=data_collator
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(test_sampler is None),
        collate_fn=data_collator
    ))
    # import ipdb; ipdb.set_trace()
    return train_loader, valid_loader

def get_valid_dataloaders(args, distributed=True):
    if args.batch_size % args.ngpus != 0:
        raise ValueError(f"Eval Batch Size for {args.batch_size} is not divisible by {args.ngpus} gpus.")

    if args.valid_dataset != "wikitext2":
        dataset = get_dataset(args.valid_dataset, "test", cache_dir=args.cache_dir, block_size=args.length)
    else:
        dataset = get_dataset(args.valid_dataset, "train", cache_dir=args.cache_dir, block_size=args.length)

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // args.ngpus,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        shuffle=(sampler is None),
        persistent_workers=True,
    )
    return dataloader