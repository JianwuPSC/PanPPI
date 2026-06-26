import pandas as pd
import numpy as np
import sys
import random
import torch
import os
from Bio import ExPASy, SwissProt
import requests
import json
import re
from collections import defaultdict

path_a = sys.argv[1] #'2711_fasta.fa'
posi_path = sys.argv[2] #'2711_eukaryotes_min900_loca_redun.txt'
report_path = sys.argv[3] #'2711_eukaryotes_report.txt'
mmseq_cluster_path = sys.argv[4] #'2711_cluster_cluster.tsv'
loca_json = sys.argv[5] #'2711_location.json'
other_source_out = sys.argv[6] #'output.txt'
signal_noise = 30
signal_noise_2 = 30


def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id=42):
    worker_seed = worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def find_mmseq_list(source_list, mmseq_dict, reverse_mmseq_dict): # source is list
    similar_source_list = []
    if len(source_list)>=1:
        for source in source_list:
            if (source in reverse_mmseq_dict.values()) :
                similar_source_list.extend(mmseq_dict[reverse_mmseq_dict.get(source)])
    return list(set(similar_source_list))

def Extract_first_domain(input_str):
    if not input_str:
        return ""
    parts = re.split(r'[,\s]+', input_str)
    
    for part in parts:
        stripped = part.strip()
        if stripped:
            return stripped
    return ""


def find_loca_list(source,sub_location_dict,reverse_local_dict): # source
    similar_source_list = []
    if source in sub_location_dict and sub_location_dict[source] != []:
       loca_list = sub_location_dict[source]
       for domian in loca_list:
           for all_name in list(set(reverse_local_dict.keys())):
               if Extract_first_domain(domian) in all_name:
                   similar_source_list.extend(reverse_local_dict[all_name])
       return similar_source_list
    else:
       return []


fix_seed(seed=42)
seed_worker(worker_id=42)

def merge_dicts(dict1, dict2):
    merged = {}

    for key, value in dict1.items():
        merged[key] = list(set(value)) if isinstance(value, list) else [value]

    for key, value in dict2.items():
        if key in merged:
            existing = merged[key]
            new_values = value if isinstance(value, list) else [value]
            merged[key] = list(set(existing + new_values))
        else:
            merged[key] = list(set(value)) if isinstance(value, list) else [value]

    return merged

###############################################################################

fa_dict = {}
with open(path_a) as fa:    
    for line in fa:
        line = line.replace('\n','')
        if not line.startswith('#'):
            if line.startswith('>'):
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                fa_dict[seq_name] += line.replace('\n','')

###############################################################################
inter_example = pd.read_table(report_path,names=['source',"target"])
report_dict1={}

for index,row in inter_example.iterrows():
    name = row['source']
    if name not in report_dict1.keys():
        report_dict1[name] = []
    report_dict1[name].append(row['target'])

inter_example = pd.read_table(report_path,names=['source',"target"])
report_dict2={}

for index,row in inter_example.iterrows():
    name = row['target']
    if name not in report_dict2.keys():
        report_dict2[name] = []
    report_dict2[name].append(row['source'])

report_dict = merge_dicts(report_dict1, report_dict2)

###############################################################################

inter_example = pd.read_table(posi_path,names=['source',"target"])
posi_dict1={}

for index,row in inter_example.iterrows():
    name = row['source']
    if name not in posi_dict1.keys():
        posi_dict1[name] = []
    posi_dict1[name].append(row['target'])

inter_example = pd.read_table(posi_path,names=['source',"target"])
posi_dict2={}

for index_number,row in inter_example.iterrows():
    name = row['target']
    if name not in posi_dict2.keys():
        posi_dict2[name] = []
    posi_dict2[name].append(row['source'])

###############################################################################

inter_example = pd.read_table(mmseq_cluster_path,names=['source',"target"])
mmseq_dict={}

for index,row in inter_example.iterrows():
    name = row['source']
    if name not in mmseq_dict.keys():
        mmseq_dict[name] = []
    mmseq_dict[name].append(row['target'])

reverse_mmseq_dict = {}
for key, value_list in mmseq_dict.items():
    for value in value_list:
        reverse_mmseq_dict[value] = key

###############################################################################

with open(loca_json, "r", encoding="utf-8") as f:
    sub_location_dict = json.load(f)

reverse_local_dict = defaultdict(list)
for key, value_list in sub_location_dict.items():
    for value in list(value_list):
        reverse_local_dict[value].append(key)

nan_sublocation =  [keys for keys,values in sub_location_dict.items() if values == []]

################################################################################

other_list1 = []
posi_list1 = []
used_dict_source = defaultdict(list)

inter_example = pd.read_table(posi_path,names=['source',"target"])
for index,row in inter_example.iterrows():
    
    source_condi = row['source'] in sub_location_dict and sub_location_dict[row['source']] != []
    target_condi = row['target'] in sub_location_dict and sub_location_dict[row['target']] != []
    
    if source_condi:
        source = row['source']
    elif not source_condi and target_condi :
        source = row['target']
    else:
        continue
    
    if source in report_dict:
        report_list = report_dict[source]
    else:
        report_list = []

    if source in posi_dict1:
        posi_list = posi_dict1[source]
    else:
        posi_list = []

    if source in posi_dict2:
        posi_list2 = posi_dict2[source]
    else:
        posi_list2 = []
        
    neg_set = list(set(list(fa_dict.keys())) - set(report_list) - set(posi_list1) - set(posi_list2) - set(used_dict_source[source]))
        
    neg_set = list(set(neg_set) - set(mmseq_dict[reverse_mmseq_dict.get(source)]) - \
                   set(find_mmseq_list(report_list, mmseq_dict, reverse_mmseq_dict)) - \
                   set(find_mmseq_list(used_dict_source[source], mmseq_dict, reverse_mmseq_dict)) - \
                   set(find_loca_list(source,sub_location_dict,reverse_local_dict)) - set(nan_sublocation))
        
    if len(neg_set) > 0 :
        target_total = random.sample(neg_set, min(20,len(neg_set)))
    for i_sample in target_total:
        target = i_sample
        used_dict_source[source].append(target)
        used_dict_source[target].append(source)
        other_list1.append('\t'.join([source,target]))
        
f=open(other_source_out,"w")
for line in other_list1:
    f.write(line+'\n')
f.close()
