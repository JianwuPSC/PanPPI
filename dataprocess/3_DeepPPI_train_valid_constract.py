df_path = '/dataset/dscript-data/pairs/new_yeast_test.csv'
df = pd.read_csv(df_path, names=['col1', 'col2', 'species', 'classify'])
df = df.dropna()
species_list = []

for idx in range(len(df)):
    seq1 = str(df.iloc[idx, 0])
    seq2 = str(df.iloc[idx, 1])
    species1 = str(df.iloc[idx, 2]).split('_')[0]
    species2 = str(df.iloc[idx, 2]).split('_')[1]
    classify = str(df.iloc[idx, 3])
    species_list.append([species1,species2,classify])

species_list
            
np.save('/path/DeepPPI/yeast_test.npy', np.array(species_list))

