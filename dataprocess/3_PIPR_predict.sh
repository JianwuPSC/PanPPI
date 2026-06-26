# PIPR
CUDA_VISIBLE_DEVICES=2 python /home/wuj/data/tools/seq_ppi/binary/model/lasagna/rcnn.py /home/wuj/data/tools/seq_ppi/yeast/preprocessed/protein.actions.tsv -1 results/yeast_wvctc_rcnn_50_5.txt 3 50 100
CUDA_VISIBLE_DEVICES=2 python rcnn.py /home/wuj/data/tools/seq_ppi/species/train_human.actions.tsv -1 result/human_wvctc_rcnn_50_5.txt 3 50 100


CUDA_VISIBLE_DEVICES=2 python /home/wuj/data/tools/seq_ppi/binary/model/lasagna/rcnn.py /home/wuj/data/tools/seq_ppi/yeast/preprocessed/protein.actions.tsv -1 results/yeast_wvctc_rcnn_50_5.txt 3 50 100
CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../multi_species/preprocessed/CeleganDrosophilaEcoli.actions.tsv -1 results/all_wvctc_rcnn_50_5.txt 3 50 100

CUDA_VISIBLE_DEVICES=2 python rcnn.py ../../../sun/preprocessed/Supp-AB.tsv -1 results/sun_wvctc_rcnn_50_5.txt 3 50 100

CUDA_VISIBLE_DEVICES=2 python zz_predict1.py /home/wuj/data/project/PLM_fine-tune/ESM2/gold_standard_dataset/species_dataset/10116/10116_posi_nega_linear_1-10.txt -1 result/human_wvctc_rcnn_50_5.txt 3 50 30
