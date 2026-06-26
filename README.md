# PanPPI
<img width="857" height="655" alt="model" src="C:/Users/Admin/Desktop/PPI_PPAM/fig/model.png" />

PanPPI: Cross-species Protein-Protein Interaction Prediction
## Description
PanPPI (Protein-Pair predictor based on Protein Language Model) is a deep learning framework that integrates a fine-tuned protein language model (PLM) with contrastive learning. It was trained on a large-scale, cross-species eukaryotic protein-protein interaction (PPI) dataset containing 3.4 million high-confidence positive pairs and actively generated negative samples. An imbalanced training strategy was employed to model the inherent sparsity of proteome-wide interactions. This framework is designed for predicting protein-protein interactions and is based on a specialized protein language model that incorporates a hybrid attention mechanism to handle both intra-protein and inter-protein residue interactions. The model achieves superior performance on PPI prediction tasks across multiple species benchmarks
## Getting started / installation
To get started using PanPPI, clone the repo:

    git clone https://github.com/JianwuPSC/PanPPI.git
    
Conda Install PanPPI

    conda env create -f environment.yaml

## Get PPI Dataset Examples

### Eukaryotic intra-species PPI dataset
This high-quality eukaryotic Protein-Protein Interaction (PPI) dataset is derived from the STRING database, which initially contained 59,309,604 proteins from 12,535 organisms. We extracted PPIs classified under eukaryotic species (1,322 species) with a high-confidence interaction score (>0.9) and required supporting evidence from experimental data or curated databases. The raw dataset of 14,030,634 PPIs underwent a rigorous refinement process: removal of reciprocal pairs and proteins shorter than 50 amino acids, resulting in 6,988,815 PPIs. Further filtering based on the availability of subcellular localization information from UniProt yielded 4,972,094 PPIs across 1,194 eukaryotic species. Finally, intra-species redundancy was reduced by clustering proteins at 50% sequence similarity and 70% coverage using MMseqs2, removing sequence-redundant PPIs. The final dataset comprises 3,436,103 non-redundant, high-confidence PPIs.
### Eukaryotic inter-species PPI dataset (positive:negative = 1:1)
1,090,278 positive samples and 1,102,151 negative samples (the imperfect 1:1 balance after redundancy removal does not impact model training or evaluation). 
The total of 2,192,429 samples was partitioned into training and validation sets (1,918,136 and 184,351 samples, respectively). 
PPIs with sequence similarity to proteins in the benchmark test datasets were filtered out. The final training set contained 1,703,981 PPIs.
### Eukaryotic inter-species PPI dataset (positive : negative = 1:5, 1:10,1:30)
The positive samples were derived from the eukaryotic cross-species PPI dataset including the validation set and one-tenth of the training set randomly selected, yielding 232,615 PPIs spanning 1,190 species. Based on this positive set, negative samples were generated at ratios of 1:5, 1:10, and 1:30, respectively. These were subsequently combined with benchmark dataset entries and subjected to redundancy reduction, resulting in final dataset sizes of 1,634,154 (1:5), 2,767,348 (1:10), and 7,134,342 (1:30) samples. The datasets were partitioned into training and test sets as follows: 1:5 (training set 1,050,844, validation set 399,622), 1:10 (training set 1,743,701, validation set 627,111), 1:30 (training set 4,187,834, validation set 1,317,910)
### Construction of test sets comprising unseen species (Cucumis sativus, Triticum aestivum, Oncorhynchus mykiss, and Rattus norvegicus)
PPI data for Cucumis sativus, Triticum aestivum, Oncorhynchus mykiss, and Rattus norvegicus were sourced from the eukaryotic intra-species PPI dataset. For each unseen species, two distinct test sets were constructed: one with a positive-to-negative ratio of 1:1 and another with a ratio of 1:10. all the data stored at the https://huggingface.co/datasets/wj5/PPLM_PPI
## Training
PanPPI incorporates a specialized fine-tuning strategy on the MINT (Multimeric INteraction Transformer) https://github.com/VarunUllanat/mint/tree/main?tab=readme-ov-file architecture. The model initialization utilizes the pre-trained weights from the MINT framework alongside the configuration file for the esm2_t33_650M_UR50D protein language model. During training, 80% of the transformer layers in the base model are frozen to preserve the pre-learned representations of protein sequences. The parameters for the embedding layer, the remaining 20% of the transformer layers, the contrastive learning head, and the protein-protein interaction (PPI) prediction classification head are re-initialized and jointly trained. The overall loss function is a weighted combination of a contrastive loss and a classification loss. A cosine annealing scheduler dynamically adjusts the contribution of each loss component: the weight of the contrastive loss decreases from 0.8 to 0.2 over the 0-10 training epochs, while the weight of the classification loss correspondingly increases from 0.2 to 0.8. This design aims to leverage unsupervised protein pair representation learning effectively during initial training stages while gradually shifting the focus towards supervised interaction prediction.

Download MINT weight file and move to raw_MINT_param

    wget https://huggingface.co/varunullanat2012/mint/blob/main/mint.ckpt

The model was trained on a balanced eukaryotic inter-species Protein-Protein Interaction (PPI) dataset, where the ratio of positive to negative samples was maintained at 1:1.

    python PanPPI_main.py
    
The trained model checkpoint for ​PPAM-PPI_1-1​ (balanced 1:1 positive-negative ratio variant) is stored at: https://huggingface.co/wj5/PPLM_PPI/blob/main/PanPPI_1-1.ckpt

##  Fine-tune PanPPI using an unbalanced sampling
PanPPI models were fine-tuned using an unbalanced sampling strategy to address proteome-wide interaction sparsity. Three specialized variants were generated with progressively increasing negative-to-positive sample ratios: PanPPI_1-5 (1:5), PanPPI_1-10 (1:10), and PanPPI_1-30 (1:30). These models are publicly available for download and inference through Hugging Face Hub at https://huggingface.co/wj5/PPLM_PPI.

    python PanPPI_fine-tuning_main.py

## Inference

The PanPPI model accepts a CSV file (Input_fasta.csv) as input for predicting protein-protein interactions. This file should contain four columns: the sequences of two proteins (e.g., MATERYI..., MKATDSS...), a unique identifier for the protein pair (e.g., ProteinA_ProteinB), and a preset classification label (where 1 indicates a known interaction and 0 indicates a non-interaction or unknown status) 
The inference script is executed with specified parameters, including the random seed and CUDA configuration for GPU acceleration. The model processes the input sequences and generates predictions in an output file named PanPPI_predict.csv.

    python PanPPI_test.py PanPPI.ckpt Input_fasta.csv 0 cuda:0 PanPPI_predict.csv
