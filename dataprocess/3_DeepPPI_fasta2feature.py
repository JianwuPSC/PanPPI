import pandas as pd
import numpy as np
import sys
sys.path.append('/path/DeepPPI/src')

from constants import *
import math
import numpy as np
import protpy
import requests
from protFeat.feature_extracter import extract_protein_feature

fa_dict = {}
with open('/path/DeepPPI/species.fa') as fa:
    for line in fa:
        line = line.replace('\n','')
        if not line.startswith('#'):
            if line.startswith('>'):
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                fa_dict[seq_name] += line.replace('\n','')

for idx in list(fa_dict.keys()):
    seq1 = fa_dict[idx]
    species1 = idx
            
    f=open('/path/DeepPPI/sample/'+species1+'.fasta',"w")
    f.write('>'+species1+'\n'+seq1+'\n')
    f.close()
            
    np.save('/path/DeepPPI/sample/'+species1+'.npy', np.array(seq1))

    extract_protein_feature("QSOrder", 0 , '/path/DeepPPI/sample', species1)
    extract_protein_feature("SOCNumber", 0,'/path/DeepPPI/sample', species1)
