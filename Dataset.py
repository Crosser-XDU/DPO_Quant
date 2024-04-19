import torch.nn as nn
import datasets
from collections import defaultdict
import tqdm
import numpy as np
import torch
from typing import Dict
from torch.utils.data import Dataset, DataLoader
import itertools
import math
from utils import add_eos

def preprocess_rlhf_dataset(row: Dict, prompt_symbol: str):
    chosen_ex = row["chosen"]
    reject_ex = row["rejected"]
    prompt_idx = chosen_ex.rfind(prompt_symbol) 
    assert prompt_idx != -1, f"Prompt not found in chosen_ex: {chosen_ex}"
    prompt = chosen_ex[:prompt_idx + len(prompt_symbol)]
    chosen_responce = chosen_ex[len(prompt):]
    reject_responce = reject_ex[len(prompt):]
    sft_responce = chosen_responce
    return prompt, chosen_responce, reject_responce, sft_responce

def get_dataset(dataset_name: str, train_test: str, cache_dir: str = None, symbol: str = None):

    print(f"Load {dataset_name} {train_test} dataset from huggingface")
    dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir, split=train_test)
    print("Done!")

    if dataset_name == "Anthropic/hh-rlhf":
        prompt_symbol = "\n\nAssistant:"
    else:
        raise ValueError(f"Unknown dataset symbol: {dataset_name}")

    data = []

    for row in tqdm.tqdm(dataset, desc="Processing dataset"):
        prompt, chosen_responce, reject_responce, sft_responce = preprocess_rlhf_dataset(row, prompt_symbol)
        data_item_dict = {'prompt': prompt, 'chosen': chosen_responce, 'rejected': reject_responce, 'sft_target': sft_responce}
        data.append(data_item_dict)
    
    return data

def tokenize_batch_element(prompt: str, chosen: str, rejected: str, tokenizer: nn.Module, max_len: int, max_prompt_len: int):
    
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    chosen_tokens = tokenizer(chosen, add_special_tokens=False)
    rejected_tokens = tokenizer(rejected, add_special_tokens=False)

    add_eos(chosen_tokens, tokenizer.eos_token_id)
    add_eos(rejected_tokens, tokenizer.eos_token_id)

    longer_response_length = max(len(chosen_tokens['input_ids']), len(rejected_tokens['input_ids']))

    if len(prompt_tokens['input_ids']) + longer_response_length > max_len:
        prompt_tokens = {k: v[-max_prompt_len:] for k, v in prompt_tokens.items()}

    # if that's still too long, truncate the response
    if len(prompt_tokens['input_ids']) + longer_response_length > max_len:
        chosen_tokens = {k: v[:max_len - max_prompt_len] for k, v in chosen_tokens.items()}
        rejected_tokens = {k: v[:max_len - max_prompt_len] for k, v in rejected_tokens.items()}

    # Concate prompt and response
    chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
    rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}


    chosen_sequence_tokens['labels'] = chosen_sequence_tokens['input_ids'][:]
    chosen_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    rejected_sequence_tokens['labels'] = rejected_sequence_tokens['input_ids'][:]
    rejected_sequence_tokens['labels'][:len(prompt_tokens['input_ids'])] = [-100] * len(prompt_tokens['input_ids'])

    batch_element = {}

    for k, toks in {'chosen': chosen_sequence_tokens, 'rejected': rejected_sequence_tokens, 'prompt': prompt_tokens}.items():
        for type_key, tokens in toks.items():
            if type_key == 'token_type_ids':
                continue
            batch_element[f'{k}_{type_key}'] = tokens

    return batch_element

def get_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        paded_batch = {}
        for key in batch[0].keys():
            if key.endswith("_input_ids") or key.endswith("_attention_mask") or key.endswith("_labels"):
                if "prompt" in key:
                    pad_elements = [torch.LongTensor(ex[key][::-1]) for ex in batch]
                else:
                    pad_elements = [torch.LongTensor(ex[key]) for ex in batch]

                if key.endswith("_input_ids"):
                    padding_value =  pad_token_id
                elif key.endswith("_attention_mask"):
                    padding_value = 0
                elif key.endswith("_labels"):
                    padding_value = -100
                else:
                    raise ValueError(f"Unknown key {key}")
                paded_batch[key] = nn.utils.rnn.pad_sequence(pad_elements, batch_first=True, padding_value=padding_value)
            else:
                paded_batch[key] = [ex[key] for ex in batch]
        return paded_batch
    return collate_fn


class CustomDataset(Dataset):
    def __init__(self, dataset_name: str, train_test: str, cache_dir: str, tokenizer: nn.Module, max_len: int, max_prompt_len: int):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_prompt_len = max_prompt_len

        self.data = get_dataset(dataset_name, train_test, cache_dir)

    def __len__(self):
        return len(self.data)

class SFTDataset(CustomDataset):
    def __init__(self, dataset_name: str, train_test: str, cache_dir: str, tokenizer: nn.Module, max_len: int, max_prompt_len: int):
        super().__init__(dataset_name, train_test, cache_dir, tokenizer, max_len, max_prompt_len)

    def __getitem__(self, idx):
        prompt = self.data[idx]['prompt']
        chosen = self.data[idx]['chosen']
        rejected = self.data[idx]['rejected']
        sft_target = self.data[idx]['sft_target']
        return tokenize_batch_element(prompt, sft_target, sft_target, self.tokenizer, self.max_len, self.max_prompt_len)
    
class DPODataset(CustomDataset):
    def __init__(self, dataset_name: str, train_test: str, cache_dir: str, tokenizer: nn.Module, max_len: int, max_prompt_len: int):
        super().__init__(dataset_name, train_test, cache_dir, tokenizer, max_len, max_prompt_len)

    def __getitem__(self, idx):
        prompt = self.data[idx]['prompt']
        chosen = self.data[idx]['chosen']
        rejected = self.data[idx]['rejected']
        sft_target = self.data[idx]['sft_target']
        return tokenize_batch_element(prompt, chosen, rejected, self.tokenizer, self.max_len, self.max_prompt_len)

def get_batch_iter(dataset_name: str,
                   tokenizer: nn.Module,
                   train_test: str,
                   batch_size: int,
                   max_len: int = 512,
                   max_prompt_len: int = 128,
                   shuffle: bool = False,
                   sft_mode: bool = False,
                   seed: int = 0,
                   cache_dir: str = None, 
                   n_examples: int = 0,
                   num_workers: int = 1,
                   ) -> iter:

    assert n_examples > batch_size, "n_examples should be greater than batch_size"

    torch.manual_seed(seed)

    if sft_mode:
        dataset = SFTDataset(dataset_name, train_test, cache_dir, tokenizer, max_len, max_prompt_len)
    else:
        dataset = DPODataset(dataset_name, train_test, cache_dir, tokenizer, max_len, max_prompt_len)

    pad_token_id = tokenizer.pad_token_id
    collate_fn = get_collate_fn(pad_token_id)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=num_workers)

    dataloader = itertools.islice(dataloader, math.ceil(n_examples / batch_size))

    return dataloader
