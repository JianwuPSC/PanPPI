import numpy as np
import pandas as pd
import logging
import os
import random
import sys
import h5py

path_a = "Eukaryotes_protein_len50_reportset.fa"
fa_dict = {}
with open(path_a) as fa:
    for line in fa:
        line = line.replace('\n','')
        if not line.startswith('#'):
            if line.startswith('>'):
                seq_name = line[1:]
                species = seq_name.split('.')[0]
                if species not in fa_dict:
                    fa_dict[species] = {}
                fa_dict[species][seq_name] = ''
            else:
                fa_dict[species][seq_name] += line.replace('\n','')


emb_path="Eukaryotes_protein_len50_reportset_fasta.h5"

def save_dict_to_hdf5(result_dict, output_path):
    with h5py.File(output_path, 'w') as hf:
        for species, proteins in result_dict.items():
            species_group = hf.create_group(species)
            for protein_name, embedding in proteins.items():
                arr = np.array([embedding], dtype=h5py.string_dtype())
                species_group.create_dataset(
                    name=protein_name,
                    data=arr,
                    compression="gzip"
                )

save_dict_to_hdf5(fa_dict, emb_path)

