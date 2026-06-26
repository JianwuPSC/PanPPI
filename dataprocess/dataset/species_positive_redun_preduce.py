from multiprocessing import Pool, Manager
from collections import defaultdict
import sys

def init_worker(shared_cluster_map, shared_filtered_signatures):
    """Initialize worker processes with shared data."""
    global g_cluster_map, g_filtered_signatures
    g_cluster_map = shared_cluster_map
    g_filtered_signatures = shared_filtered_signatures

def process_chunk(pairs_chunk):
    """Process a chunk of protein pairs and filter by unique cluster combinations."""
    local_filtered_pairs = set()
    for a, b in pairs_chunk:
        cluster_a = g_cluster_map.get(a)
        cluster_b = g_cluster_map.get(b)
        if not cluster_a or not cluster_b:
            continue  # Skip pairs without clustering information
        
        # Generate unique cluster combination signature (ordered)
        signature = frozenset({cluster_a, cluster_b})
        
        # Check if the same cluster combination already exists (Manager.dict() operations are thread-safe)
        if signature not in g_filtered_signatures:
            g_filtered_signatures[signature] = True  # Mark as processed
            local_filtered_pairs.add((a, b))
    
    return local_filtered_pairs

def parallel_filter_pairs(pairs, cluster_map, processes=32, chunk_size=10000):
    """Parallel filtering of protein pairs based on unique cluster combinations."""
    # Create shared data
    with Manager() as manager:
        shared_cluster_map = manager.dict(cluster_map)
        shared_filtered_signatures = manager.dict()  # Store processed cluster combinations
        
        # Split data into chunks
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        
        # Start parallel processing
        with Pool(processes=processes, 
                 initializer=init_worker, 
                 initargs=(shared_cluster_map, shared_filtered_signatures)) as pool:
            results = pool.map(process_chunk, chunks)
        
        # Merge results
        return set().union(*results)

if __name__ == '__main__':
    # Parse command line arguments
    cluster_path = sys.argv[1] # Eukaryotes_protein_len50_cluster.tsv
    raw_pair_path = sys.argv[2] # Eukaryotes_positive_report_min900_loca.txt
    filter_pair_path = sys.argv[3] # Eukaryotes_positive_report_min900_loca_redund1.txt
    
    # Load cluster data (protein: cluster_id)
    cluster_map = {}  
    with open(cluster_path) as f:
        for line in f:
            cluster, protein = line.strip().split()
            cluster_map[protein] = cluster

    # Load raw protein pairs
    pairs = set()
    with open(raw_pair_path) as f:
        for line in f:
            a, b, classify = line.strip().split()
            if a in cluster_map and b in cluster_map:
                pairs.add(tuple(sorted((a, b))))
    
    # Apply parallel filtering
    filtered_pairs = parallel_filter_pairs(list(pairs), cluster_map, processes=32)
   
    # Write filtered pairs to output file
    with open(filter_pair_path, "w") as f:
        for a, b in sorted(filtered_pairs):
            f.write(f"{a}\t{b}\n")
