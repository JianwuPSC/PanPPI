#!/bin/bash
# =============================================================================
# species_split.sh – Template for building a species‑specific PPI dataset
#                    as described in the PPAM‑PPI paper.
#
# This script processes one eukaryotic species at a time, applying:
#   1) Subcellular localisation filtering of positive pairs
#   2) Protein sequence extraction & clustering (MMseqs2)
#   3) Intra‑species redundancy reduction based on sequence clusters
#   4) Active negative sampling
#   5) Construction of balanced/imbalanced positive‑negative datasets
#   6) (Optional) Cross‑species redundancy removal – handled later globally
#
# Required external tools:
#   - MMseqs2 (for clustering)
#   - Perl + custom extract_fasta.pl script (provided by the authors)
#   - Python 3 with modules: pandas, numpy, torch, Bio, requests, etc.
#
# Required input files (per species, adjust paths accordingly):
#   - ${species}_eukaryotes_high‑confidence.txt   : STRING high‑confidence PPIs
#   - ${species}_location.json                    : Subcellular localisation from UniProt
#   - ${species}_fasta.fa                         : Protein sequences (to be created)
#   - ${species}_eukaryotes_report.txt            : Full STRING interaction report
#   - ${species}_cluster_cluster.tsv              : MMseqs2 clustering output
#
# Output files (written to designated directories):
#   - *_eukaryotes_min900_loca.txt                : Localisation‑filtered positives
#   - *_eukaryotes_min900_loca_redun.txt          : Redundancy‑reduced positives
#   - *_negative_loca_1‑30.txt                    : Actively generated negatives (1:30 ratio)
#   - *_posi_nega_*.txt                           : Final training/validation/test splits
# =============================================================================

# ---------------------------- Configuration ----------------------------------
# Species identifier (e.g. 9606 for human, 4932 for yeast, 3702 for Arabidopsis)
species=9606

# Paths to input data (modify to match your directory structure)
base_data_dir="/path/to/string_data"
output_dir="/path/to/output/${species}"
mkdir -p ${output_dir}

# Input files (adjust names as needed)
high_conf_ppi="${base_data_dir}/${species}_eukaryotes_high‑confidence.txt"
location_json="${base_data_dir}/${species}_location.json"
string_report="${base_data_dir}/${species}_eukaryotes_report.txt"
fasta_raw="${base_data_dir}/Eukaryotes_protein_len50.fa"   # multi‑species FASTA

# Output files (within species‑specific output directory)
posi_loca="${output_dir}/${species}_eukaryotes_min900_loca.txt"
posi_redun="${output_dir}/${species}_eukaryotes_min900_loca_redun.txt"
neg_1_30="${output_dir}/${species}_negative_loca_1-30.txt"
posi_nega_1_1="${output_dir}/${species}_posi_nega_1-1.txt"
posi_nega_1_10="${output_dir}/${species}_posi_nega_1-10.txt"
train_out="${output_dir}/train.csv"
valid_out="${output_dir}/valid.csv"
test_out="${output_dir}/test.csv"
cross_out="${output_dir}/cross.csv"

# Temporary files
fasta_species="${output_dir}/${species}_fasta.fa"
cluster_tsv="${output_dir}/${species}_cluster_cluster.tsv"
tmp_dir="${output_dir}/tmp_mmseqs"

# -----------------------------------------------------------------------------

echo "=============================================="
echo "Processing species: ${species}"
echo "=============================================="

# ------------------- Step 1: Filter positives by localisation -----------------
# Keep only PPIs where at least one protein has a known subcellular location.
# This is required for subsequent active negative sampling.
echo "[Step 1] Filtering positive pairs with subcellular localisation info"
python species_positive_exist_location.py \
    "${high_conf_ppi}" \
    "${location_json}" \
    "${posi_loca}"

# ------------------- Step 2: Extract species‑specific FASTA ------------------
# From a multi‑species FASTA file (e.g. Eukaryotes_protein_len50.fa), extract
# only the sequences belonging to this species. The extract_fasta.pl script
# expects a list of protein IDs (one per line) and the full FASTA.
# The list of protein IDs is typically obtained from the PPI file.
echo "[Step 2] Extracting protein sequences for species ${species}"
cut -f1,2 "${posi_loca}" | tr '\t' '\n' | sort -u > "${output_dir}/${species}_fasta.list"
perl ~/data/project/ppi_predict/dataprocess/extract_fasta.pl \
    "${fasta_raw}" \
    "${output_dir}/${species}_fasta.list" \
    "${fasta_species}"

# ------------------- Step 3: Sequence clustering with MMseqs2 ----------------
# Cluster proteins at 50% sequence identity and 70% coverage. The resulting
# clusters are used for redundancy reduction.
echo "[Step 3] Running MMseqs2 clustering (50% identity, 70% coverage)"
mmseqs easy-cluster "${fasta_species}" \
    "${output_dir}/${species}_cluster" \
    "${tmp_dir}" \
    --min-seq-id 0.5 -c 0.7 --threads 16

# Rename cluster TSV to expected name (source: cluster representative, target: member)
mv "${output_dir}/${species}_cluster_cluster.tsv" "${cluster_tsv}"

# ------------------- Step 4: Intra‑species redundancy reduction --------------
# Remove redundant positive pairs: if a pair (A,B) shares clusters with an
# already selected pair (C,D) (i.e. A with C, B with D in same clusters),
# then (A,B) is discarded.
echo "[Step 4] Removing sequence‑redundant positive pairs within species"
python species_positive_redun_preduce.py \
    "${cluster_tsv}" \
    "${posi_loca}" \
    "${posi_redun}"

# ------------------- Step 5: Active negative sampling ------------------------
# Generate negative pairs at a 30:1 negative:positive ratio (configurable).
# The script uses localisation, homology, co‑expression and transferred scores
# to ensure high‑quality non‑interacting pairs.
echo "[Step 5] Generating active negative samples (30× negatives)"
python species_negative_produce.py \
    "${fasta_species}" \
    "${posi_redun}" \
    "${string_report}" \
    "${cluster_tsv}" \
    "${location_json}" \
    "${neg_1_30}"

# ------------------- Step 6: Build balanced & imbalanced datasets ------------
# Combine positive and negative pairs to create final training/validation/test
# splits with specified positive:negative ratios. The species_posi_nega_constract.py
# script merges positives with a subset of negatives.
# For 1:1 ratio (balanced)
echo "[Step 6a] Constructing 1:1 positive‑negative dataset"
python species_posi_nega_constract.py \
    "${posi_redun}" \
    "${location_json}" \
    "${neg_1_30}" \
    "${posi_nega_1_1}" \
    1

# For 1:10 ratio (used for genome‑wide screening simulations)
echo "[Step 6b] Constructing 1:10 positive‑negative dataset"
python species_posi_nega_constract.py \
    "${posi_redun}" \
    "${location_json}" \
    "${neg_1_30}" \
    "${posi_nega_1_10}" \
    10

# ------------------- Step 7: Graph‑based train/valid/test split -------------
# Use network community detection (Louvain) to partition the interaction graph
# such that proteins in different splits do not share edges (no leakage).
# The script species_posi_nega_constract_train_valid_test.py produces
# train.csv, valid.csv, test.csv, and cross.csv (pairs that cross splits).
echo "[Step 7] Splitting dataset into non‑overlapping train/valid/test sets"
python species_posi_nega_constract_train_valid_test.py \
    "${posi_nega_1_1}" \
    "${train_out}" \
    "${valid_out}" \
    "${test_out}" \
    "${cross_out}"

# ------------------- Step 8: (Optional) Cross‑species redundancy -------------
# After processing all species, a global script (not included here) merges
# all species‑specific datasets and applies sequence‑based redundancy reduction
# across species. That step is described in the paper and should be run
# separately after all species have been processed.

echo "=============================================="
echo "Finished processing species ${species}"
echo "Output files are in ${output_dir}"
echo "=============================================="
