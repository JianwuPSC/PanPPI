
####### conda activate dplm

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import sys
import random

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'

fasta_csv_path = sys.argv[1]
output_path = sys.argv[2]
device = sys.argv[3] # cuda:0

##################################

from transformers import T5Tokenizer, T5EncoderModel
import torch
import re

T5_tokenizer = T5Tokenizer.from_pretrained('/path/ProtTrans/param', do_lower_case=False)
T5_model = T5EncoderModel.from_pretrained("/path/ProtTrans/param")

def T5_embeding(T5_model, T5_tokenizer, sequence_examples, device):
    
    T5_model = T5_model.to(device)
    sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in [sequence_examples]]
    # tokenize sequences and pad up to the longest sequence in the batch
    ids = T5_tokenizer(sequence_examples, add_special_tokens=True, padding="longest")

    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    # generate embeddings
    with torch.no_grad():
        embedding_repr = T5_model(input_ids=input_ids, attention_mask=attention_mask)

    embedding_out = embedding_repr.last_hidden_state[0,:,:].mean(dim=0)
    return embedding_out.cpu().detach().numpy()

#######################################

##### T5_embedding

name_list=[]
represent_list=[]
class_list=[]

esm_represent_source_list=[]
esm_represent_target_list=[]

df = pd.read_csv(fasta_csv_path, names=['col1', 'col2', 'species', 'classify'])
for index,row in df.iterrows():

    source = row['col1'] # traget P53691
    target =  row['col2'] # source P02829
    species = row['species']
    ppi_class = row['classify']

    if len(source) > 1024:
        start = random.randint(0, len(source) - 1024 + 1)
        source = source[start : start + 1024]
    if len(target) > 1024:
        start = random.randint(0, len(target) - 1024 + 1)
        target = target[start : start + 1024]

    source_seq_repre = T5_embeding(T5_model, T5_tokenizer, source, device)
    target_seq_repre = T5_embeding(T5_model, T5_tokenizer, target, device)
                   
    name_list.append(species)
    class_list.append(ppi_class)
    
    esm_represent_source_list.append(source_seq_repre)
    esm_represent_target_list.append(target_seq_repre)

    del source_seq_repre; del target_seq_repre
    torch.cuda.empty_cache()

    print(index)

################################################################

from sklearn.model_selection import train_test_split

name_df = pd.DataFrame(name_list)
name_df = name_df.rename(columns={0: 'name'})
represent_df = pd.DataFrame(represent_list)

esm_represent_source_df = pd.DataFrame(esm_represent_source_list)
esm_represent_target_df = pd.DataFrame(esm_represent_target_list)

class_df = pd.DataFrame(class_list)
class_df = class_df.rename(columns={0: 'class'})

esm_represent_soucer_df = pd.concat([name_df,esm_represent_source_df,esm_represent_target_df,class_df],axis=1)
esm_represent_soucer_df.to_csv(output_path+'/'+'T5_represent_df.csv', index=False,header=False)

