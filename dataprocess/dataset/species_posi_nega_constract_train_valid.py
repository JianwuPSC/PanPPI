import pandas as pd
import sys
import math

def Negative_sample_split(posi_path, neg_source_path):
    """
    Split positive and negative protein-protein interaction samples.
    Reads positive and negative interaction files and formats them into lists.
    
    Args:
        posi_path (str): Path to the file containing positive interaction pairs
        neg_source_path (str): Path to the file containing negative interaction pairs
    
    Returns:
        tuple: Two lists containing formatted positive and negative interaction strings
    """
    
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
    
    inter_example = pd.read_table(neg_source_path,names=['source',"target"])
    neg_source_dict={}
    other_source_list = []
    
    for index,row in inter_example.iterrows():
        name = row['source']
        if name not in neg_source_dict.keys():
            neg_source_dict[name] = []
        neg_source_dict[name].append(row['target'])
    
    for key, value in neg_source_dict.items():
        for target in neg_source_dict[key]:
            name = '\t'.join([key,target,'0'])
            other_source_list.append(name)
            
    return posi_list, other_source_list

#############################################################################

def Graph_constract(df,split_threshold=2,resolution=2):
    """
    Construct and partition a protein-protein interaction network for dataset splitting.
    Uses graph theory and community detection to create non-overlapping train/validation/test sets.
    
    Args:
        df (DataFrame): DataFrame with protein interaction pairs (protein1, protein2, label)
        split_threshold (int): Maximum component size before applying Louvain community detection
        resolution (float): Resolution parameter for Louvain community detection algorithm
    
    Returns:
        tuple: Three DataFrames for training, validation, and cross-set interactions
    """

    import networkx as nx
    import numpy as np
    # Build a protein-protein interaction network
    G = nx.Graph()
    G.add_edges_from(zip(df["protein1"], df["protein2"]))
    
    # Get connected components and sort by size in descending order
    connected_components = sorted(nx.connected_components(G), key=len, reverse=True)

    # Define split threshold (components larger than this will be split)
    split_threshold = split_threshold #max(min(0.01 * len(G.nodes()),50),5)
    large_components = [c for c in connected_components if len(c) > split_threshold]
    small_components = [c for c in connected_components if len(c) <= split_threshold]

    import community as community_louvain  # Requires python-louvain package
    # Split each large connected component into sub-communities
    split_groups = []
    for comp in large_components:
        subgraph = G.subgraph(comp)
        partition = community_louvain.best_partition(subgraph, resolution=resolution)  # Adjust resolution for split granularity
        split_groups.extend([{k for k,v in partition.items() if v==i} for i in set(partition.values())])

    # Combine all groups to be allocated (split large groups + original small groups)
    all_components = split_groups + small_components

    from heapq import heappush, heappop
    # Initialize target ratios
    target_ratios = {'train': 0.7, 'valid': 0.3}
    current_sizes = {'train': 0, 'valid': 0}
    assigned_proteins = set()
    proteins_train = set()
    proteins_valid = set()
    proteins_test = set ()

    # Create a max-heap based on component size (prioritize large connected components)
    heap = []
    for comp in all_components:
        heappush(heap, (-len(comp), comp))  # Use negative numbers to simulate max-heap

    # Allocation logic
    while heap:
        _, comp = heappop(heap)
        available_sets = [s for s in target_ratios if current_sizes[s]/(sum(current_sizes.values())+1e-8) < target_ratios[s]]
    
        if not available_sets:  # All sets have reached target ratio
            assigned_proteins.update(comp)
            continue
        
        # Allocate to the set with the lowest current ratio
        target_set = min(available_sets, key=lambda x: current_sizes[x]/target_ratios[x])
        current_sizes[target_set] += len(comp)
        assigned_proteins.update(comp)
    
        # Record allocation result
        if target_set == 'train':
            proteins_train.update(comp)
        elif target_set == 'valid':
            proteins_valid.update(comp)
        #else:
        #    proteins_test.update(comp)

    # Allocate interaction pairs
    train_df = df[df["protein1"].isin(proteins_train) & df["protein2"].isin(proteins_train)]
    valid_df = df[df["protein1"].isin(proteins_valid) & df["protein2"].isin(proteins_valid)]
    #test_df = df[df["protein1"].isin(proteins_test) & df["protein2"].isin(proteins_test)]

    cross_pairs_mask = ~df.index.isin(train_df.index) & \
                       ~df.index.isin(valid_df.index)
                       #~df.index.isin(test_df.index)
                   
    cross_df = df[cross_pairs_mask]
    # Verify isolation
    assert set(proteins_train).isdisjoint(proteins_valid)
    #assert set(proteins_train).isdisjoint(proteins_test)
    #assert set(proteins_valid).isdisjoint(proteins_test)
    # Calculate actual ratios
    total = len(train_df) + len(valid_df)
    print(f"train: {len(train_df)/total:.1%} (train: 70%)")
    print(f"valid: {len(valid_df)/total:.1%} (valid: 30%)")

    return train_df, valid_df, cross_df

# Command line argument parsing
posi_path = sys.argv[1] #"2711_eukaryotes_min900_redun.txt"

train_out = sys.argv[2] # h5_embedding/train.csv
valid_out = sys.argv[3] # h5_embedding/valid.csv
cross_out = sys.argv[4] # h5_embedding/cross.csv

df  = pd.read_table(posi_path,names=['protein1',"protein2","label"])

# Partition the dataset using graph-based method
train_df, valid_df, cross_df = Graph_constract(df,split_threshold=5,resolution=2)

# Save the partitioned datasets
train_df.to_csv(train_out, index=False,header=False)
valid_df.to_csv(valid_out, index=False,header=False)
cross_df.to_csv(cross_out, index=False,header=False)
