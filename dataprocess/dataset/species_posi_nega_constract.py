import pandas as pd
import sys
import math
import numpy as np
import random
import torch
import os
from Bio import ExPASy, SwissProt
import requests
import json
import re
from collections import defaultdict

def Negative_sample_split(posi_path,loca_json,neg_source_path,nega_count):

    ###############################################
    ########## positive pair list
    ###############################################
    inter_example = pd.read_table(posi_path,names=['source',"target"])
    posi_dict={}
    posi_list=[]
    for index,row in inter_example.iterrows():
        name = row['source']
        if name not in posi_dict.keys():
            posi_dict[name] = []
        posi_dict[name].append(row['target'])

    for key, value in posi_dict.items():
        for target in posi_dict[key]:
            name = '\t'.join([target,key,'1'])
            posi_list.append(name)

    ################################################
    ########### location json
    ################################################
    with open(loca_json, "r", encoding="utf-8") as f:
        sub_location_dict = json.load(f)

    reverse_local_dict = defaultdict(list)
    for key, value_list in sub_location_dict.items():
        for value in list(value_list):
            reverse_local_dict[value].append(key)

    nan_sublocation =  [keys for keys,values in sub_location_dict.items() if values == []]
    
    ################################################
    ########### negative pair list
    ################################################ 
    inter_example = pd.read_table(neg_source_path,names=['source',"target"])
    neg_source_dict={}
    other_source_list = []

    for index,row in inter_example.iterrows():
        name = row['source']
        if name not in neg_source_dict.keys():
            neg_source_dict[name] = []
        neg_source_dict[name].append(row['target'])

    other_list1 = []
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
        
        neg_set = list(set(list(neg_source_dict[source])) - set(used_dict_source[source]))
        
        if len(neg_set) > 0 :
            target_total = random.sample(neg_set, min(int(nega_count),len(neg_set)))
        for i_sample in target_total:
            target = i_sample
            used_dict_source[source].append(target)
            used_dict_source[target].append(source)
            other_list1.append('\t'.join([source,target,'0']))

    return posi_list, other_list1

########################################################################################

posi_path = sys.argv[1] #"2711_eukaryotes_min900_redun.txt"
loca_json = sys.argv[2] # 2711_location.json
neg_source_path = sys.argv[3] #"2711_negative_source.txt"
posi_nega_output = sys.argv[4] #output
nega_count = sys.argv[5] #[1,5,10,30]

posi_list, other_source_list = Negative_sample_split(posi_path,loca_json,neg_source_path,nega_count)

total_list = posi_list + other_source_list
split_data = [row.split('\t') for row in total_list]

f=open(posi_nega_output,"w")
for line in total_list:
    f.write(line+'\n')
f.close()
