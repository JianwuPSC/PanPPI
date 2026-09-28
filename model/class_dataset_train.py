import pandas as pd
import numpy as np
import sys
import os
import re
import h5py
import random
import itertools
import pickle
import shutil
import math
from typing import List, Sequence, Tuple, Union,TypeVar, Iterable
import logging
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, BatchSampler, Sampler
import torch.nn.init as init
import torch.nn.functional as F


proteinseq_toks = {'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']}

class Alphabet(object):
    def __init__(
        self,
        standard_toks: Sequence[str],
        prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
        append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
        prepend_bos: bool = True,
        append_eos: bool = False,
        use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)
        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}
        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ["<eos>", "<unk>", "<pad>", "<cls>", "<mask>"]
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def to_dict(self):
        return self.tok_to_idx.copy()

    def get_batch_converter(self, truncation_seq_length: int = None):
        if self.use_msa:
            return MSABatchConverter(self, truncation_seq_length)
        else:
            return BatchConverter(self, truncation_seq_length)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        if name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        elif name in ("MSA Transformer", "msa_transformer"):
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = False
            use_msa = True
        elif "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks["toks"]
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

    def _tokenize(self, text) -> str:
        return text.split()

    def tokenize(self, text, **kwargs) -> List[str]:
        """
        Inspired by https://github.com/huggingface/transformers/blob/master/src/transformers/tokenization_utils.py
        Converts a string in a sequence of tokens, using the tokenizer.
        Args:
            text (:obj:`str`):
                The sequence to be encoded.
        Returns:
            :obj:`List[str]`: The list of tokens.
        """

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                # We strip left and right by default
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]
    
#########################################################################################

class Only_Seqs_CSVDataset(Dataset):
    def __init__(self, df_path, col1, col2, species, classify):
        super().__init__()
        self.df = pd.read_csv(df_path, names=[col1, col2, species, classify])
        self.df = self.df.dropna()
        self.seqs1 = self.df[col1].tolist()
        self.seqs2 = self.df[col2].tolist()
        self.species = self.df[species].tolist()
        self.classify = self.df[classify].tolist()

    def __len__(self):
        return len(self.seqs1)

    def __getitem__(self, index):
        return self.seqs1[index], self.seqs2[index], self.species[index], self.classify[index]


class Seqs_CSVDataset(Dataset):
    def __init__(self, df_path, col1, col2, species, classify, embedding_path):
        super().__init__()
        self.df = pd.read_csv(df_path, names=[col1, col2, species, classify])
        self.df = self.df.dropna()
        self.embedding_path = embedding_path
        self.fa_dict = self._h5_fasta(self.embedding_path)
        self.valid_indices = self._precompute_valid_indices(self.fa_dict)

    def _precompute_valid_indices(self, fa_dict):

        valid_indices = []
        for idx in range(len(self.df)):
            seq1 = str(self.df.iloc[idx, 0])
            seq2 = str(self.df.iloc[idx, 1])
            species = str(self.df.iloc[idx, 2])
            
            try:
                if (''.join(seq1.split('.')[1:]) in fa_dict or seq1 in fa_dict) and (''.join(seq2.split('.')[1:]) in fa_dict or seq2 in fa_dict):
                    valid_indices.append(idx)
            except:
                continue

        return valid_indices

    def _h5_fasta(self, embedding_path):
        
        fa_dict = {}
        with open(embedding_path) as fa:
            for line in fa:
                line = line.replace('\n','')
                if not line.startswith('#'):
                    if line.startswith('>'):
                        seq_name = line[1:]
                        fa_dict[seq_name] = ''
                    else:
                        fa_dict[seq_name] += line.replace('\n','')

        return fa_dict
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, index):
        # 通过预计算的valid_indices获取真实索引
        real_idx = self.valid_indices[index]
        #print(self.valid_indices)
        seq1 = str(self.df.iloc[real_idx, 0])
        seq2 = str(self.df.iloc[real_idx, 1])
        species = str(self.df.iloc[real_idx, 2])
        label = self.df.iloc[real_idx, 3]
        if seq1 in self.fa_dict:
            fasta1 = self.fa_dict[seq1]
            fasta2 = self.fa_dict[seq2]
        else :
            fasta1 = self.fa_dict[''.join(seq1.split('.')[1:])]
            fasta2 = self.fa_dict[''.join(seq2.split('.')[1:])]

        return fasta1, fasta2, species, label


def h5_fasta(string_dataset,species,name):
    with h5py.File(string_dataset, 'r') as f:    
        if name in f[species].keys():
            # ['metadata', 'species'] # [embeddings   proteins]
            embeddings = list(f[species][name][:])
            embeddings = [s.decode('utf-8') for s in embeddings][0]            
    return embeddings


class Class_CSVDataset(Dataset):
    def __init__(self, df_path, col1, col2, species, classify,length1, length2, embedding_path):
        super().__init__()
        self.df = pd.read_csv(df_path, names=[col1, col2, species, classify, length1,length2])
        self.df = self.df.dropna()
        # 路径设置
        self.embedding_path = embedding_path
        self.valid_indices = self._precompute_valid_indices(embedding_path)
        self.length1 = self.df[length1].tolist()
        self.length2 = self.df[length2].tolist()
        

    def _precompute_valid_indices(self, embedding_path):

        valid_indices = []
        for idx in range(len(self.df)):
            seq1 = str(self.df.iloc[idx, 0])
            seq2 = str(self.df.iloc[idx, 1])
            species = str(self.df.iloc[idx, 2])
            
            try:
                with h5py.File(embedding_path, 'r') as f:
                    if species in f and seq1 in f[species] and seq2 in f[species]:
                        valid_indices.append(idx)
            except:
                continue
                
        return valid_indices
    
    def _h5_fasta(self, string_dataset, species, name):
        
        with h5py.File(string_dataset, 'r') as f:    
            if species in f and name in f[species]:
                # ['metadata', 'species'] # [embeddings   proteins]
                embeddings = list(f[species][name][:])
                embeddings = [s.decode('utf-8') for s in embeddings][0]
                
        return embeddings

    def __len__(self):
        return len(self.valid_indices)
    
    def lens_merge(self, c):
        
        length1, length2 = c
        if length1 > 512:
            length1 = 512
        else:
            length1 = length1
        if length2 > 512:
            length2 = 512
        else:
            length2 = length2        
        return (length1+length2)/2
        
    
    def get_metadata_lens(self):
        
        length1_arr = np.array(self.length1)
        length2_arr = np.array(self.length2)
        

        aa = list(zip(list(length1_arr[self.valid_indices]), list(length2_arr[self.valid_indices])))

        length_array = [self.lens_merge(c) for i, c in enumerate(aa)]
        return length_array
    

    def __getitem__(self, index):
        # 通过预计算的valid_indices获取真实索引
        real_idx = self.valid_indices[index]
        #print(self.valid_indices)
        seq1 = str(self.df.iloc[real_idx, 0])
        seq2 = str(self.df.iloc[real_idx, 1])
        species = str(self.df.iloc[real_idx, 2])
        label = self.df.iloc[real_idx, 3]
        
        fasta1 = h5_fasta(self.embedding_path, species, seq1)
        fasta2 = h5_fasta(self.embedding_path, species, seq2)
        
        return fasta1, fasta2, species, label

#############################################################################################

class Class_CollateFn:
    def __init__(self,truncation_seq_length=None):
        self.alphabet = Alphabet.from_architecture('ESM-1b')
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):
        chains = zip(*batches)
        chains = [self.convert(c) for i,c in enumerate(chains) if i <2]
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]
        classify = torch.tensor([cl[3] for i, cl in enumerate(batches)]).unsqueeze(-1)
        class_id = [classify.repeat(1,c.shape[1]) for i, c in enumerate(chains)]
        chains = torch.cat(chains, -1)
        chain_ids = torch.cat(chain_ids, -1)
        class_id = torch.cat(class_id, -1)
        chains, chain_ids = self.symmetric_input(chains, chain_ids)
        return chains, chain_ids, class_id # chains_tokens [encode], chain_ids [0,0,0,1,1,1,1]

    def convert(self, seq_str_list):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
            for seq_str in seq_str_list
        ]
        if self.truncation_seq_length:
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                    seq_encoded_list[i] = seq[start : start + self.truncation_seq_length]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        if self.truncation_seq_length:
            assert max_len <= self.truncation_seq_length
        tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)

        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, : len(seq_encoded)] = seq
        return tokens

    def symmetric_input(self, chains, chain_ids):

        if not torch.is_grad_enabled():
            return chains, chain_ids
    
        batch_size, seq_len = chain_ids.shape

        swap_decisions = torch.rand(batch_size) < 0.5
        if not swap_decisions.any():
            return chains, chain_ids
    
        processed_chains = chains.clone()
        processed_chain_ids = chain_ids.clone()
    
        for b in range(batch_size):
            if swap_decisions[b]:
                sample_chain_ids = chain_ids[b]

                chain0_mask = (sample_chain_ids == 0)
                chain1_mask = (sample_chain_ids == 1)
            
                chain0_len = chain0_mask.sum().item()
                chain1_len = chain1_mask.sum().item()
            
                if chain0_len > 0 and chain1_len > 0:
                    chain0_indices = torch.where(chain0_mask)[0]
                    chain1_indices = torch.where(chain1_mask)[0]
             
                    new_order = torch.cat([chain1_indices, chain0_indices])   
                    processed_chains[b,:] = processed_chains[b,new_order]
                
                    new_chain_ids = torch.cat([
                        torch.zeros(chain1_len, dtype=chain_ids.dtype),
                        torch.ones(chain0_len, dtype=chain_ids.dtype)
                    ])
                    processed_chain_ids[b, :] = torch.cat([
                        new_chain_ids,
                        torch.zeros(seq_len - chain0_len - chain1_len, 
                        dtype=chain_ids.dtype)
                    ])
    
        return processed_chains, processed_chain_ids


class SortishSampler(Sampler): #
    """Returns indices such that inputs with similar lengths are close together."""

    def __init__(
        self, sequence_lengths: Iterable, bucket_size: int, num_replicas: int = 1, rank: int = 0
    ):

        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [
            self.data[i * bucket_size : i * bucket_size + bucket_size] for i in range(n_buckets)
        ]
        self.rank = rank
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        np.random.seed(self.epoch)
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        
#########################################################################################

class ApproxBatchSampler(BatchSampler):
    """
    Parameters:
    -----------
    sampler : Pytorch Sampler
            Choose base sampler class to use for bucketing
    max_tokens : int
            Maximum number of tokens per batch
    max_batch: int
            Maximum batch size
    sample_lengths : array-like
            List of lengths of sequences in the order of the dataset
    """

    def __init__(
        self,
        sampler,
        max_tokens,
        max_batch,
        sample_lengths,
        max_square_tokens=np.inf,
        drop_last=False,
        batch_size=None,
        max_len=512
    ):
        super().__init__(sampler, max_batch, drop_last)
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.max_len = max_len
        self.batches = self._build_batches()
        
    def _build_batches(self):
        batches = []
        length = 0
        ell_sq = 0
        batch = []
        for i, idx in enumerate(self.sampler):
            this_length = min(self.max_len, self.sample_lengths[idx])
            linear = (len(batch) + 1) * max(length, this_length)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length**2)
            if linear <= self.max_tokens and quadratic < self.max_square_tokens:
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length**2)
                if len(batch) == self.max_batch:
                    batches.append(batch)
                    batch = []
                    length = 0
            else:
                if len(batch) == 0:
                    print('Current batch is empty! idx is ', idx)
                    continue
                batches.append(batch)
                batch = [idx]
                length = this_length
                ell_sq = this_length**2
        if len(batch) > 0:
            batches.append(batch)
            
        if self.sampler.num_replicas > 1:
            num_samples = torch.tensor(len(batches)).cuda()
            print(f'==============Local Rank {self.sampler.rank} Num Samples {num_samples}==============')
            dist.all_reduce(num_samples, op=dist.ReduceOp.MAX)
            print(f'==============All Reduce Num Samples {num_samples}==============')
            num_samples = num_samples.item()

            if len(batches) < num_samples:
                # padding_size = num_samples - len(batches)
                a = num_samples // len(batches)
                b = num_samples % len(batches)
                new_batches = batches * a
                new_batches += batches[:b]
                assert len(new_batches) == num_samples
                batches = new_batches
            print(f'==============After Reduce, Rank{self.sampler.rank}, Num Samples {num_samples}==============')
        return batches
            
    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch


############################################################################

def setup_dataloader(
        ds,  
        max_tokens=10000, bucket_size=50, 
        max_batch_size=32,  num_workers=0,
        rank=0, world_size=1,
        mini_run=False,
        max_len=2048,
    ) -> DataLoader:
    collater = Class_CollateFn(truncation_seq_length=max_len) # batch: input_ids, input_mask, targets
    if mini_run:
        dl = DataLoader(dataset=ds, shuffle=False, batch_size=max_batch_size, num_workers=num_workers, collate_fn=collater) # dataload 方式
    else:
        lens = ds.get_metadata_lens()
        #print(lens)
        
        train_sortish_sampler = SortishSampler(lens, bucket_size, num_replicas=world_size, rank=rank) 
        train_sampler = ApproxBatchSampler(train_sortish_sampler, max_tokens, max_batch_size, lens, max_len=max_len)
        dl = DataLoader(dataset=ds, batch_sampler=train_sampler, num_workers=num_workers, collate_fn=collater,shuffle=False,drop_last=False)
    return dl
