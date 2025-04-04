#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import List, Optional, Dict, Any, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoProcessor
from PIL import Image
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return len(next(iter(self.encodings.values())))


def get_pileval(tokenizer: PreTrainedTokenizer, nsamples: int, seqlen: int, device: Optional[str], seed: int = 0) -> DataLoader[torch.Tensor]:
    dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation").shuffle(seed=seed)

    samples = []
    n_run = 0
    for data in dataset:
        line_encoded = tokenizer.encode(data["text"].strip())
        if 0 < len(line_encoded) <= seqlen:
            sample = torch.tensor([line_encoded], device=device)
            samples.append(sample)
            n_run += 1
        if n_run == nsamples:
            break

    # Concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // seqlen

    # Create training dataset by splitting concatenated samples
    train_dataset = [cat_samples[:, i * seqlen:(i + 1) * seqlen] for i in range(n_split)]

    # Create batched samples
    batch_inps = torch.cat(train_dataset, dim=0)

    return batch_inps

def get_mlperf_data(data_path: str,
                   tokenizer: AutoTokenizer = None,
                   batch_size: int = 1,
                   num_calib_data: int = 128,
                   seqlen: int = 2048,
                   device: str = 'cpu') -> DataLoader[torch.Tensor]:
    
    import pickle

    print("mlperf calibration data path: ", data_path)

    with open(data_path, 'rb') as fh:
        mlperf_df = pickle.load(fh)

    system_prompt_instruction = mlperf_df['input'].tolist()[:num_calib_data]

    batch_encoded = tokenizer.batch_encode_plus(
        system_prompt_instruction,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=seqlen,
    )
    if device:
        batch_encoded = batch_encoded.to(device)

    tokenized_dataset = CustomDataset({"input_ids": batch_encoded["input_ids"]})

    calib_dataloader: DataLoader[List[Dict[str, torch.Tensor]]] = DataLoader(tokenized_dataset, batch_size=batch_size,
                                                                             shuffle=False, drop_last=True)  # type: ignore

    return calib_dataloader


def get_wikitext2(tokenizer: PreTrainedTokenizer,
                  nsamples: int,
                  seqlen: int,
                  device: Optional[str],
                  seed: int = 0) -> DataLoader[torch.Tensor]:

    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    trainenc = trainenc.to(device)

    import random
    random.seed(seed)
    torch.random.manual_seed(seed)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({'input_ids': inp, 'attention_mask': attention_mask})
    batch_inps = torch.cat([sample['input_ids'] for sample in traindataset], dim=0)
    return batch_inps


def get_calib_dataloader_for_benchmark(dataset_name: str = "pileval_for_awq_benchmark",
                                       tokenizer: AutoTokenizer = None,
                                       batch_size: int = 1,
                                       num_calib_data: int = 128,
                                       seqlen: int = 2048,
                                       device: str = 'cpu') -> DataLoader[torch.Tensor]:
    if dataset_name == "pileval_for_awq_benchmark":
        samples = get_pileval(tokenizer, num_calib_data, seqlen, device, seed=42)
        if batch_size != len(samples):
            print(f"[INFO-Warning] For AWQ benchmark, batch_size should be {len(samples)}. Changing batch_size to {len(samples)}.")
            batch_size = len(samples)
    elif dataset_name == "wikitext_for_gptq_benchmark":
        samples = get_wikitext2(tokenizer, num_calib_data, seqlen, device)
    else:
        raise NotImplementedError

    calib_dataloader: DataLoader[List[Dict[str, torch.Tensor]]] = DataLoader(samples, batch_size=batch_size,
                                                                             shuffle=False, drop_last=True)  # type: ignore

    return calib_dataloader


def get_calib_dataloader_to_tensor(dataset_name: str = "cnn_dailymail",
                                   tokenizer: AutoTokenizer = None,
                                   batch_size: int = 1,
                                   num_calib_data: int = 512,
                                   seqlen: int = 512,
                                   shuffle: bool = False,
                                   device: Optional[str] = None) -> DataLoader[torch.Tensor]:
    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        text_data = dataset["text"][:num_calib_data]
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        text_data = dataset["article"][:num_calib_data]
    elif dataset_name == "wikitext":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        text_data = dataset["text"][:num_calib_data]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer(text_data, return_tensors="pt", padding=True, truncation=True, max_length=seqlen)
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=shuffle, drop_last=True)

    return calib_dataloader


def get_calib_dataloader_to_dict(dataset_name: str = "cnn_dailymail",
                                 tokenizer: AutoTokenizer = None,
                                 batch_size: int = 1,
                                 num_calib_data: int = 512,
                                 seqlen: int = 512,
                                 device: Optional[str] = None) -> DataLoader[Dict[str, torch.Tensor]]:

    def make_data_block(examples: Dict[str, List[str]],
                        tokenizer: AutoTokenizer = None,
                        prompt_col_name: str = '',
                        max_length: int = 512) -> dict[str, List[List[torch.Tensor]]]:
        res: dict[str, List[List[torch.Tensor]]] = tokenizer(examples[prompt_col_name],
                                                             padding=True,
                                                             truncation=True,
                                                             max_length=max_length)
        return res

    def my_collate_fn(blocks: List[Dict[str, List[List[str]]]]) -> Dict[str, torch.Tensor]:
        data_batch = {}
        data_batch["input_ids"] = torch.Tensor([block["input_ids"] for block in blocks])
        if device:
            data_batch["input_ids"] = data_batch["input_ids"].to(device)
        return data_batch

    if dataset_name == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        prompt_col_name = "text"
    elif dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        prompt_col_name = "article"
    elif dataset_name == "wikitext":
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        prompt_col_name = "text"
    else:
        raise NotImplementedError

    dataset = dataset.select(
        indices=[i for i in range(min(len(dataset), num_calib_data))],
        keep_in_memory=True,
    )
    tokenized_datasets = dataset.map(make_data_block,
                                     batched=True,
                                     batch_size=len(dataset),
                                     num_proc=1,
                                     remove_columns=dataset.column_names,
                                     keep_in_memory=True,
                                     fn_kwargs={
                                         'tokenizer': tokenizer,
                                         'prompt_col_name': prompt_col_name,
                                         'max_length': seqlen
                                     })

    calib_dataloader = DataLoader(tokenized_datasets, batch_size=batch_size, collate_fn=my_collate_fn)

    return calib_dataloader

def get_ultrachat(dataset_name: str = "HuggingFaceH4/ultrachat_200k",
                  tokenizer: AutoTokenizer = None,
                  batch_size: int = 1,
                  num_calib_data: int = 512,
                  seqlen: int = 512,
                  device: Optional[str] = None) -> DataLoader[List[Dict[str, torch.Tensor]]]:
    MAX_SEQUENCE_LENGTH = seqlen

    ds = load_dataset(dataset_name, split="train_sft")
    ds = ds.shuffle(seed=42).select(range(num_calib_data))

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,)}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"],
                         padding=False,
                         max_length=MAX_SEQUENCE_LENGTH,
                         truncation=True,
                         add_special_tokens=False,)

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    traindataset = []
    for i in range(len(ds['input_ids'])):
        inp = torch.tensor([ds['input_ids'][i]], device=device)
        attention_mask = torch.tensor([ds['attention_mask'][i]], device=device)
        traindataset.append({'input_ids': inp, 'attention_mask': attention_mask})

    calib_dataloader: DataLoader[List[Dict[str, torch.Tensor]]] = DataLoader(traindataset, batch_size=None,
                                                                             shuffle=False)  # type: ignore
    return calib_dataloader

def get_calib_dataloader(
        dataset_name: str, processor: AutoProcessor = None, **kwargs: Any
) -> Union[DataLoader[torch.Tensor], DataLoader[List[Dict[str, torch.Tensor]]], DataLoader[Dict[str, torch.Tensor]]]:
    if dataset_name in ["pileval", "cnn_dailymail", "wikitext"]:
        return get_calib_dataloader_to_tensor(dataset_name, **kwargs)
    elif dataset_name in ["pileval_for_awq_benchmark", "wikitext_for_gptq_benchmark"]:
        return get_calib_dataloader_for_benchmark(dataset_name, **kwargs)
    elif "ultrachat" in dataset_name:
        return get_ultrachat(dataset_name, **kwargs)
    elif "ScienceQA" in dataset_name:
        return get_scienceqa("ScienceQA_VAL", processor=processor, **kwargs)
    elif dataset_name.endswith('pkl'):
        return get_mlperf_data(dataset_name, **kwargs)
    else:
        raise NotImplementedError

def get_dataset(path, name, subset, tokenizer, seqlen):
    text = load_dataset(path=path, name=name, split=subset)
    if path in ['wikitext']:
        strtext = "\n\n".join(text['text'])
    elif path in ['shibing624/AdvertiseGen']:
        strtext = "\n\n".join(str(i[0]) + str(i[1]) for i in list(zip(list(text['content']), list(text['summary']))))
    tokenized_text = tokenizer(strtext, return_tensors='pt')
    tokenized_text_len = tokenized_text.input_ids.shape[1]

    sample = []
    for i in range(0, tokenized_text_len - seqlen - 1, seqlen):
        sample.append(tokenized_text.input_ids[:, i : i + seqlen])
    sample = torch.dstack(sample).squeeze(0).permute(1, 0)

    dataset = TensorDataset(sample)
    return dataset

def get_loader(path, name, subset, tokenizer, seqlen=1024, num_batch=-1, batch_size=1, shuffle=False):
    dataset = get_dataset(path, name, subset, tokenizer, seqlen)
    data_size = len(dataset)

    if num_batch != -1:  # num_batch == -1 using the whole dataset
        sample_size = min(data_size, num_batch * batch_size)
        subset_indices = torch.randperm(data_size)[:sample_size]
        dataset = Subset(dataset, subset_indices)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

# for VLM
def image_parser(image_file, sep=','):
    out = image_file.split(sep)
    return out

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out

def get_scienceqa(dataset_name: str = "ScienceQA_VAL",
                  processor: AutoProcessor = None,
                  tokenizer: AutoTokenizer = None,
                  batch_size: int = 1,
                  num_calib_data: int = 512,
                  seqlen: int = 512,
                  device: Optional[str] = None) -> DataLoader[List[Dict[str, torch.Tensor]]]:

    from vlmeval.dataset import build_dataset
    from transformers.feature_extraction_utils import BatchFeature

    dataset = build_dataset(dataset_name)
    data = dataset.data
    data_indices = [i for i in data['index']]
    lt = min(num_calib_data, len(dataset))

    traindataset = []
    for i in tqdm(range(lt)):
        # prompt, image
        message = dataset.build_prompt(data.iloc[i])  # data: question, answer, hint, image, etc.
        prompt, image_path = message_to_promptimg(message, dataset=dataset_name)
        image = Image.open(image_path)

        # inputs
        messages = [
            {'role': 'user', 'content': [
                {'type': 'image'},
                {'type': 'text', 'text': prompt}
            ]}
        ]
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(image, input_text, return_tensors='pt').to(device)

        traindataset.append(inputs)

    calib_dataloader: DataLoader[List[BatchFeature]] = DataLoader(traindataset, batch_size=None, shuffle=False)
    return calib_dataloader

def message_to_promptimg(message, dataset=None):
    num_images = len([x for x in message if x['type'] == 'image'])
    if num_images == 0:
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        image = None
    else:
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        images = [x['value'] for x in message if x['type'] == 'image']
        if 'BLINK' == dataset:
            image = concat_images_vlmeval(images, target_size=512)
        else:
            image = images[0]
    return prompt, image
