import pandas as pd
import numpy as np
import sys
import random
import torch
import os
from Bio import ExPASy, SwissProt
import requests
import json
from collections import defaultdict

# Parse command line arguments
posi_path = sys.argv[1] #'2711_eukaryotes_min900_redun.txt'
loca_json = sys.argv[2] #'2711_location.json'
other_posi_out = sys.argv[3] #'zz_posi.txt'

# Load subcellular localization JSON data
with open(loca_json, "r", encoding="utf-8") as f:
    sub_location_dict = json.load(f)

# Create a reverse dictionary mapping localization terms to proteins
reverse_local_dict = defaultdict(list)
for key, value_list in sub_location_dict.items():
    for value in list(value_list):
        reverse_local_dict[value].append(key)

# Initialize list for storing filtered positive interactions
posi_list1 = []
used_dict_source = defaultdict(list)

# Read positive interaction data
inter_example = pd.read_table(posi_path,names=['source',"target"])
# Iterate through each interaction pair
for index,row in inter_example.iterrows():
    
    # Check if either source or target protein has subcellular localization data
    source_condi = row['source'] in sub_location_dict and sub_location_dict[row['source']] != []
    target_condi = row['target'] in sub_location_dict and sub_location_dict[row['target']] != []
    
    # Include interaction if at least one protein has localization data
    if source_condi or target_condi:
        posi_list1.append('\t'.join([row['source'],row['target']]))
    else:
        continue

# Write filtered interactions to output file
f=open(other_posi_out,"w")
for line in posi_list1:
    f.write(line+'\n')
f.close()
