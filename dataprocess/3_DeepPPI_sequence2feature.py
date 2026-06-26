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

df_path = '/dataset/dscript-data/pairs/new_fly_test.csv'
df = pd.read_csv(df_path, names=['col1', 'col2', 'species', 'classify'])
df = df.dropna()

for idx in range(len(df)):
    seq1 = str(df.iloc[idx, 0])
    seq2 = str(df.iloc[idx, 1])
    species1 = str(df.iloc[idx, 2]).split('_')[0]
    species2 = str(df.iloc[idx, 2]).split('_')[1]
    classify = str(df.iloc[idx, 3])
            
    f=open('/path/DeepPPI/sample/'+species1+'.fasta',"w")
    f.write('>'+species1+'\n'+seq1+'\n')
    f.close()
            
    f=open('/path/DeepPPI/sample/'+species2+'.fasta',"w")
    f.write('>'+species2+'\n'+seq2+'\n')
    f.close()
            
    np.save('/path/DeepPPI/sample/'+species1+'.npy', np.array(seq1))
    np.save('/path/DeepPPI/sample/'+species2+'.npy', np.array(seq2))
            
    extract_protein_feature("QSOrder", 0 , '/path/DeepPPI/sample', species1)
    extract_protein_feature("SOCNumber", 0,'/path/DeepPPI/sample', species1)
    extract_protein_feature("QSOrder", 0 , '/path/DeepPPI/sample', species2)
    extract_protein_feature("SOCNumber", 0,'/path/DeepPPI/sample', species2)
