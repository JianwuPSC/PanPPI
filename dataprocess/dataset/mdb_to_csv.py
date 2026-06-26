import lmdb
import pickle
import pandas as pd
from tqdm import tqdm
import os

def parse_lmdb_dataset(lmdb_path, csv_path):
    """
    Parse a protein-protein interaction LMDB dataset and save as CSV and Parquet files.
    The dataset contains serialized sample objects with protein sequences and interaction labels.
    
    Args:
        lmdb_path (str): Path to the LMDB database
        csv_path (str): Path to save the output CSV file
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    
    data_records = []
    
    with env.begin() as txn:

        num_samples = txn.stat()['entries']
        print(f"include {num_samples} samples")
        
        pbar = tqdm(total=num_samples, desc="parse samples")
        
        # Iterate through all key-value pairs
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                sample = (pickle.loads(value))
                if type(sample) == dict:
                    
                    record = {
                        'id': key.decode('utf-8'),
                        'protein1_sequence': sample.get('primary_1', ''),
                        'protein2_sequence': sample.get('primary_2', ''),
                        'protein1_length': sample.get('protein_length_1', 0),
                        'protein2_length': sample.get('protein_length_2', 0),
                        'interaction': sample.get('interaction', 0)
                        }
                
                    data_records.append(record)
                    pbar.update(1)
                
            except (pickle.UnpicklingError, KeyError) as e:
                print(f"error (ID: {key}): {e}")
                continue
        pbar.close()
    
    df = pd.DataFrame(data_records)
    
    df.to_csv(csv_path, index=False)
    print(f"save csv: {csv_path}")
    
    return df

if __name__ == "__main__":
    lmdb_path = "/home/wuj/data/tools/SaProt/example/HumanPPI/normal/valid"
    csv_path = "human_dataset/huamn_valid.csv"
    
    df = parse_lmdb_dataset(lmdb_path, csv_path)
    
    print("\nhead file:")
    print(df.head())


###################################################################################################

import lmdb
import json
import re
from tqdm import tqdm

def robust_lmdb_parser(lmdb_path):
    """
    Robust LMDB parser that handles malformed JSON data.
    This function reads an LMDB database where values are stored as JSON strings,
    attempts multiple parsing strategies to handle formatting issues.
    
    Args:
        lmdb_path (str): Path to the LMDB database
        
    Returns:
        list: List containing all parsed records
    """
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    results = []
    
    with env.begin() as txn:
        # Get total number of entries
        length = int(txn.get(b'length').decode())
        
        # Iterate through all entries
        for i in range(length):
            key = str(i).encode()
            value = txn.get(key)
            
            if not value:
                print(f"Warning: Key {i} has no value")
                continue
                
            try:
                # Attempt direct parsing
                data = json.loads(value)
            except json.JSONDecodeError:
                try:
                    # Attempt parsing after preprocessing
                    value_str = value.decode('utf-8')
                    value_str = value_str.strip().rstrip('\n')
                    
                    # Fix common issues
                    if value_str.startswith('b\'') and value_str.endswith('\''):
                        value_str = value_str[2:-1]
                    
                    # Attempt parsing again
                    data = json.loads(value_str)
                except:
                    # Ultimate solution: manual JSON repair
                    try:
                        data = manual_json_repair(value)
                    except Exception as e:
                        print(f"Failed to parse record {i}: {str(e)}")
                        data = {'error': f"Parse failed: {str(e)}", 'raw_data': value.decode('utf-8', errors='replace')}
            
            results.append(data)
    
    return results

# Usage example
lmdb_path = '/path/SaProt/example/HumanPPI/normal/test/'

parsed_records = robust_lmdb_parser(lmdb_path)
df = pd.DataFrame(parsed_records)
df.to_csv('dataset/human_dataset/huamn_test.csv', index=False)
