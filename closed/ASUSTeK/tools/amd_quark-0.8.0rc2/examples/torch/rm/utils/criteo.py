#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
implementation of criteo dataset
"""

# pylint: disable=unused-argument,missing-docstring

import logging
import os
import random
import numpy as np
# pytorch
import torch
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("criteo")

import data_loader_terabyte

class Criteo(Dataset):

    def __init__(self,
                 data_path,
                 name,
                 pre_process,
                 use_cache,
                 count=None,
                 samples_to_aggregate_fix=None,
                 samples_to_aggregate_min=None,
                 samples_to_aggregate_max=None,
                 samples_to_aggregate_quantile_file=None,
                 samples_to_aggregate_trace_file=None,
                 test_num_workers=0,
                 max_ind_range=-1,
                 sub_sample_rate=0.0,
                 mlperf_bin_loader=False,
                 randomize="total",
                 memory_map=False):
        super().__init__()

        self.count = count
        self.random_offsets = []
        self.use_fixed_size = ((samples_to_aggregate_quantile_file is None) and
                               (samples_to_aggregate_min is None or samples_to_aggregate_max is None))

        if self.use_fixed_size:
            # fixed size queries
            self.samples_to_aggregate = 1 if samples_to_aggregate_fix is None else samples_to_aggregate_fix
            self.samples_to_aggregate_min = None
            self.samples_to_aggregate_max = None
        else:
            # variable size queries
            self.samples_to_aggregate = 1
            self.samples_to_aggregate_min = samples_to_aggregate_min
            self.samples_to_aggregate_max = samples_to_aggregate_max
            self.samples_to_aggregate_quantile_file = samples_to_aggregate_quantile_file

        if name == "kaggle":
            raw_data_file = data_path + "/train.txt"
            processed_data_file = data_path + "/kaggleAdDisplayChallenge_processed.npz"
        elif name == "multihot-criteo":
            raw_data_file = data_path + "/day"

        self.use_mlperf_bin_loader = mlperf_bin_loader and memory_map and name == "multihot-criteo"
        # debug prints
        # print("dataset filenames", raw_data_file, processed_data_file)
        if self.use_mlperf_bin_loader:
            test_file = data_path + "/terabyte_processed_test_v2_dense.bin"

            if not os.path.exists(test_file):
                data_loader_terabyte.numpy_to_binary(
                    input_files=[raw_data_file + "_23_sparse_multi_hot.npz", raw_data_file + "_23_dense.npy", raw_data_file + "_23_labels.npy",],
                    output_file_path=data_path,
                    split="test")

            self.test_data = data_loader_terabyte.CriteoBinDataset(
                data_file=data_path,
                batch_size=self.samples_to_aggregate,
                max_ind_range=max_ind_range,
            )

            self.test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
            )
        # WARNING: Note that the orignal dataset returns number of samples, while the
        # binary dataset returns the number of batches. Therefore, when using a mini-batch
        # of size samples_to_aggregate as an item we need to adjust the original dataset item_count.
        # On the other hand, data loader always returns number of batches.
        self.num_individual_samples = len(self.test_data)
        if self.use_fixed_size:
            # the offsets for fixed query size will be generated on-the-fly later on
            print("Using fixed query size: " + str(self.samples_to_aggregate))
            if self.use_mlperf_bin_loader:
                self.num_aggregated_samples = len(self.test_data)
                # self.num_aggregated_samples2 = len(self.test_loader)
            else:
                self.num_aggregated_samples = (self.num_individual_samples + self.samples_to_aggregate - 1) // self.samples_to_aggregate
                # self.num_aggregated_samples2 = len(self.test_loader)
        else:
            # the offsets for variable query sizes will be pre-generated here
            if self.samples_to_aggregate_quantile_file is None:
                # generate number of samples in a query from a uniform(min,max) distribution
                print("Using variable query size: uniform distribution (" + str(self.samples_to_aggregate_min) + "," + str(self.samples_to_aggregate_max) + ")")
                done = False
                qo = 0
                while done is False:
                    self.random_offsets.append(int(qo))
                    qs = random.randint(self.samples_to_aggregate_min, self.samples_to_aggregate_max)
                    qo = min(qo + qs, self.num_individual_samples)
                    if qo >= self.num_individual_samples:
                        done = True
                self.random_offsets.append(int(qo))

                # compute min and max number of samples
                nas_max = (self.num_individual_samples + self.samples_to_aggregate_min - 1) // self.samples_to_aggregate_min
                nas_min = (self.num_individual_samples + self.samples_to_aggregate_max - 1) // self.samples_to_aggregate_max
            else:
                # generate number of samples in a query from a custom distribution,
                # with quantile (inverse of its cdf) given in the file. Note that
                # quantile is related to the concept of percentile in statistics.
                #
                # For instance, assume that we have the following distribution for query length
                # length = [100, 200, 300,  400,  500,  600,  700] # x
                # pdf =    [0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05] # p(x)
                # cdf =    [0.1, 0.7, 0.8, 0.85,  0.9, 0.95,  1.0] # f(x) = prefix-sum of p(x)
                # The inverse of its cdf with granularity of 0.05 can be written as
                # quantile_p = [.05, .10, .15, .20, .25, .30, .35, .40, .45, .50, .55, .60, .65, .70, .75, .80, .85, .90, .95, 1.0] # p
                # quantile_x = [100, 100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 300, 300, 400, 500, 600, 700] # q(p) = x, such that f(x) >= p
                # Notice that once we have quantile, we can apply inverse transform sampling method.
                print("Using variable query size: custom distribution (file " + str(samples_to_aggregate_quantile_file) + ")")
                with open(self.samples_to_aggregate_quantile_file, 'r') as f:
                    line = f.readline()
                    quantile = np.fromstring(line, dtype=int, sep=", ")
                # debug prints
                # print(quantile)
                # print(len(quantile))

                l = len(quantile)
                done = False
                qo = 0
                while done is False:
                    self.random_offsets.append(int(qo))
                    pr = np.random.randint(low=0, high=l)
                    qs = quantile[pr]
                    qo = min(qo + qs, self.num_individual_samples)
                    if qo >= self.num_individual_samples:
                        done = True
                self.random_offsets.append(int(qo))

                # compute min and max number of samples
                nas_max = (self.num_individual_samples + quantile[0] - 1) // quantile[0]
                nas_min = (self.num_individual_samples + quantile[-1] - 1) // quantile[-1]

            # reset num_aggregated_samples
            self.num_aggregated_samples = len(self.random_offsets) - 1

            # check num_aggregated_samples
            if self.num_aggregated_samples < nas_min or nas_max < self.num_aggregated_samples:
                raise ValueError("Sannity check failed")

        # limit number of items to count if needed
        if self.count is not None:
            self.num_aggregated_samples = min(self.count, self.num_aggregated_samples)

        # dump the trace of aggregated samples
        if samples_to_aggregate_trace_file is not None:
            with open(samples_to_aggregate_trace_file, 'w') as f:
                for l in range(self.num_aggregated_samples):
                    if self.use_fixed_size:
                        s = l * self.samples_to_aggregate
                        e = min((l + 1) * self.samples_to_aggregate, self.num_individual_samples)
                    else:
                        s = self.random_offsets[l]
                        e = self.random_offsets[l + 1]
                    f.write(str(s) + ", " + str(e) + ", " + str(e - s) + "\n")

    def get_item_count(self):
        # get number of items in the dataset
        return self.num_aggregated_samples

    ''' lg compatibilty routine '''
    def unload_query_samples(self):
        self.items_in_memory = {}
        self.items_in_memory_len = {}

    def count_query_samples_len(self, sample_list):
        self.items_in_memory_len = {}
        self.sample_lens_list = []
        for l in sample_list:
            if self.use_fixed_size:
                s = l * self.samples_to_aggregate
                e = min((l + 1) * self.samples_to_aggregate, self.num_individual_samples)
            else:
                s = self.random_offsets[l]
                e = self.random_offsets[l + 1]
            sample_len = e - s
            self.items_in_memory_len[l] = sample_len
            if sample_len not in self.sample_lens_list:
                self.sample_lens_list.append(sample_len)

    def mlperf_bin_load_query_samples(self, sample_list):
        self.items_in_memory = {}
        for l in sample_list:
            if self.use_fixed_size:
                s = l * self.samples_to_aggregate
                e = min((l + 1) * self.samples_to_aggregate, self.num_individual_samples)
            else:
                s = self.random_offsets[l]
                e = self.random_offsets[l + 1]
            sample_len = e - s
            X, lS_i, T = self.test_data.read(s, sample_len)
            lS_i = lS_i.contiguous()
            T = T.contiguous()
            self.items_in_memory[l] = (X, lS_i, T)

    ''' lg compatibilty routine '''
    def load_query_samples(self, sample_list):
        self.items_in_memory = {}
        self.items_in_memory_len = {}
        self.sample_lens_list = []

        # WARNING: notice that while DataLoader is iterable-style, the Dataset
        # can be iterable- or map-style, and Criteo[Bin]Dataset are the latter
        # This means that we can not index into DataLoader, but can enumerate it,
        # while we can index into the dataset itself.
        for l in sample_list:
            # approach 1: single sample as an item
            '''
            self.items_in_memory[l] = self.test_data[l]
            '''
            # approach 2: multiple samples as an item
            if self.use_fixed_size:
                s = l * self.samples_to_aggregate
                e = min((l + 1) * self.samples_to_aggregate, self.num_individual_samples)
            else:
                s = self.random_offsets[l]
                e = self.random_offsets[l + 1]

            sample_len = e - s
            self.items_in_memory_len[l] = sample_len
            if sample_len not in self.sample_lens_list:
                self.sample_lens_list.append(sample_len)

            if self.use_mlperf_bin_loader:
                # NOTE: in binary dataset the values are transformed
                # ls_t = list(zip(*ls))
                # X = torch.cat(ls_t[0])
                # (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
                # lS_o = torch.stack([torch.tensor(range(len_ls)) for _ in range(num_s)])
                # lS_i = torch.cat(ls_t[2], dim=1)
                # T = torch.cat(ls_t[3])
                X, lS_i, T = self.test_data.read(s, sample_len)
                self.items_in_memory[l] = (X, lS_i, T)
            else:
                ls = [self.test_data[i] for i in range(s, e)]
                # NOTE: in original dataset the values are not transformed
                # and collate besides stacking them also transforms them
                self.items_in_memory[l] = self.test_loader.collate_fn(ls)


    ''' lg compatibilty routine '''
    def get_samples(self, id_list):
        # build list tuples as need by the batch conversion routine
        # index i from id_list corresponds to a particular query_id
        ls = []
        for i in id_list:
            ls.append(self.items_in_memory[i])

        # approach 1: collate a mini-batch of single samples
        '''
        if self.use_mlperf_bin_loader:
            # NOTE: in binary dataset the values are transformed
            ls_t = list(zip(*ls))

            X = torch.cat(ls_t[0])
            (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
            lS_o = torch.stack([torch.tensor(range(len_ls)) for _ in range(num_s)])
            lS_i = torch.cat(ls_t[2], dim=1)
            T = torch.cat(ls_t[3])
        else:
            # NOTE: in original dataset the values are not transformed and collate besides stacking transforms them
            X, lS_o, lS_i, T = self.test_loader.collate_fn(ls)
        '''
        # approach 2: collate a mini-batch of multiple samples
        # NOTE: recall that the samples have already been transformed for both datasets
        # (by earlier calls in load_query_samples), therefore we just need to stack them
        ls_t = list(zip(*ls))
        X = torch.cat(ls_t[0])
        lS_i = torch.cat(ls_t[2], dim=1)
        if self.use_mlperf_bin_loader:
            (num_s, len_ls) = lS_i.size()
        else:
            (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
        lS_o = torch.LongTensor(num_s, len_ls)
        for i in range(num_s):
            lS_o[i] = torch.arange(len_ls)
        T = torch.cat(ls_t[3])
        return (X, lS_o, lS_i, T)


# Pre  processing
def pre_process_criteo_dlrm(x):
    return x

def get_dataset(args):
    kwargs = {"randomize": 'total', "memory_map": True}
    # dataset to use
    # --count-samples can be used to limit the number of samples used for testing

    ds = Criteo(
        data_path=args.dataset_path,
        name=args.dataset,
        pre_process=pre_process_criteo_dlrm,  # currently an identity function
        use_cache=args.cache,  # currently not used
        count=args.count_samples,
        samples_to_aggregate_fix=args.samples_to_aggregate_fix,
        samples_to_aggregate_min=args.samples_to_aggregate_min,
        samples_to_aggregate_max=args.samples_to_aggregate_max,
        samples_to_aggregate_quantile_file=args.samples_to_aggregate_quantile_file,
        samples_to_aggregate_trace_file=args.samples_to_aggregate_trace_file,
        test_num_workers=0,
        max_ind_range=args.max_ind_range,
        sub_sample_rate=args.data_sub_sample_rate,
        mlperf_bin_loader=True,
        **kwargs
    )
    return ds

# Post processing
class DlrmPostProcess:
    def __init__(self):
        self.good = 0
        self.total = 0
        self.roc_auc = 0
        self.results = []

    def __call__(self, results, expected=None, result_dict=None):
        processed_results = []
        n = len(results)
        for idx in range(0, n):
            # NOTE: copy from GPU to CPU while post processing, if needed. Alternatively,
            # we could do this on the output of predict function in backend_pytorch_native.py
            result = results[idx].detach().cpu()
            target = expected[idx]
            processed_results.append([result, target])
            # debug prints
            # print(result)
            # print(expected[idx])

            # accuracy metric
            if result.round() == target:
                self.good += 1
        self.total += n
        return processed_results

    def add_results(self, results):
        self.results.append(results)

    def start(self):
        self.good = 0
        self.total = 0
        self.roc_auc = 0
        self.results = []

    def finalize(self, result_dict, ds=False, output_dir=None):
        # AUC metric
        from intel_pytorch_extension import core
        self.results = np.concatenate(self.results, axis=0)
        results, targets = list(zip(*self.results))
        results = np.array(results)
        targets = np.array(targets)
        self.roc_auc, _, _ = core.roc_auc_score(torch.from_numpy(targets).reshape(-1), torch.from_numpy(results).reshape(-1))

        result_dict["good"] = self.good
        result_dict["total"] = self.total
        result_dict["roc_auc"] = self.roc_auc

SUPPORTED_DATASETS = {
    "debug": (Criteo, pre_process_criteo_dlrm, DlrmPostProcess(), {"randomize": 'total', "memory_map": True}),
    "multihot-criteo-sample": (Criteo, pre_process_criteo_dlrm, DlrmPostProcess(), {"randomize": 'total', "memory_map": True}),
    "multihot-criteo": (Criteo, pre_process_criteo_dlrm, DlrmPostProcess(), {"randomize": 'total', "memory_map": True}),
}
