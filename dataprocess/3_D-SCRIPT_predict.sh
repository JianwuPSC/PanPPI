# dplm
python /path/D-SCRIPT/dscript/commands/embed.py --seqs=dataset/dscript-data/seqs/yeast.fasta -o yeast.h5
python /path/D-SCRIPT/dscript/commands/eval.py --model=dataset/dscript-data/models/dscript_human_v2.pt --test=dataset/dscript-data/pairs/yeast_test.tsv --embedding=yeast.h5 -o=yeast_predict.txt
