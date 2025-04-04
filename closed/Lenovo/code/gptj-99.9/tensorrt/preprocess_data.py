#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#           http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to preprocess the data for BERT."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from code.common import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset

G_INST_TEMPLATE = "Summarize the following news article:"
G_GPTJ6B_MAX_INPUT_SEQLEN = 1919

G_PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)


def prepare_tokenizer(checkpoint_path, padding_side="left"):
    """
    Prepare the tokenizer for the cnn dailymail
    """
    logging.info(f"Initializing tokenizer from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        model_max_length=G_GPTJ6B_MAX_INPUT_SEQLEN,
        padding_side=padding_side,
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def preprocess_cnndailymail_prompt(data):
    sources = [G_PROMPT_INPUT.format_map(
        example) for example in data]
    targets = [f"{example['output']}" for example in data]

    return sources, targets


def preprocess_cnndailymail_gptj6b(data_dir, model_dir, preprocessed_data_dir):
    cnn_val_json_path = os.path.join(
        data_dir, "cnn-daily-mail", "cnn_eval.json")
    output_dir = os.path.join(preprocessed_data_dir,
                              "cnn_dailymail_tokenized_gptj")
    ckpt_path = os.path.join(model_dir, "GPTJ-6B", "checkpoint-final")
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Creating GPT tokenizer...")
    tokenizer = prepare_tokenizer(ckpt_path, padding_side="right")
    logging.info("Done creating tokenizer.")

    logging.info("Reading CNN dailymail examples...")
    with open(cnn_val_json_path, 'r') as fh:
        data = json.load(fh)
    sources, targets = preprocess_cnndailymail_prompt(data)
    logging.info(
        f"Loaded {len(sources)} samples from {cnn_val_json_path}")
    data_len = len(sources)
    logging.info(f"Done reading {data_len} CNN dailymail examples.")

    # Converting input strings to tokenized id.
    # 6/7/2023: Note that TRT-LLM has "masked_tokens" and "attention mask" which are opposite to each other at the moment
    # All inputs will be padded to 1919
    logging.info(f"Converting {data_len} articles to tokens...")
    input_batch = tokenizer.batch_encode_plus(
        sources, return_tensors="pt",
        padding='max_length', truncation=True,
        max_length=G_GPTJ6B_MAX_INPUT_SEQLEN
    )

    input_ids = input_batch.input_ids.numpy().astype(np.int32)
    attention_mask = input_batch.attention_mask.numpy().astype(np.int32)
    masked_tokens = 1 - attention_mask
    input_real_seqlen = np.sum(attention_mask, axis=1).astype(np.int32)
    print(
        f"Shape check: input_id: {input_ids.shape} attention_mask: {attention_mask.shape} input_lengths: {input_real_seqlen.shape}")
    logging.info("Done converting articles to tokens.")

    logging.info(
        f"Saving tokenized id, masks, and input lengths to {output_dir} ...")
    np.save(os.path.join(output_dir, "input_ids_padded.npy"), input_ids)
    np.save(os.path.join(output_dir, "attention_mask.npy"), attention_mask)
    np.save(os.path.join(output_dir, "masked_tokens.npy"), masked_tokens)
    np.save(os.path.join(output_dir, "input_lengths.npy"), input_real_seqlen)

    logging.info("Done saving preprocessed data.")


def preprocess_calibration(data_dir, model_dir, preprocessed_data_dir):
    # Preprocess dataset based on https://github.com/mlcommons/inference/blob/master/language/gpt-j/prepare-calibration.py
    calib_list = Path("build/inference/language/gpt-j/calibration-list.txt")
    assert calib_list.exists(), f"Calibration list {calib_list} doesn't exist!"

    dataset = load_dataset("cnn_dailymail", name="3.0.0", split='train')
    train = dict((x['id'], x) for x in dataset)

    with open(calib_list, 'r') as fid:
        calibration_ids = fid.read().splitlines()

    inputs = []
    for id in calibration_ids:
        calibration_sample = train[id]
        x = dict()
        x["instruction"] = G_INST_TEMPLATE
        x["input"] = calibration_sample["article"]
        x["output"] = calibration_sample["highlights"]
        inputs.append(x)

    sources, _ = preprocess_cnndailymail_prompt(inputs)
    sources = [{"text": row} for row in sources]
    assert len(sources) == 1000, "The length of the calibration list is not 1000!"

    hf_dataset = Dataset.from_list(sources)
    # Cannot have "cnn_dailymail" in the path because of TRTLLM quantization rule.
    dataset_dir = Path(preprocessed_data_dir) / "gptj" / "mlperf_gptj_openorca_calibration_1k"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    hf_dataset.to_parquet(dataset_dir / "data.parquet")

    logging.info(f"Finished processing calibration dataset at {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir", "-d",
        help="Directory containing the input data.",
        default="build/data"
    )
    parser.add_argument(
        "--model_dir", "-m",
        help="Directory containing the models.",
        default="build/models"
    )
    parser.add_argument(
        "--preprocessed_data_dir", "-o",
        help="Output directory for the preprocessed data.",
        default="build/preprocessed_data"
    )
    args = parser.parse_args()
    data_dir = args.data_dir
    model_dir = args.model_dir
    preprocessed_data_dir = args.preprocessed_data_dir

    preprocess_cnndailymail_gptj6b(data_dir, model_dir, preprocessed_data_dir)
    preprocess_calibration(data_dir, model_dir, preprocessed_data_dir)

    print("Done!")


if __name__ == '__main__':
    main()
