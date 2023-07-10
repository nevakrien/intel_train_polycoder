KEY ISSUES:

1. preplexity calculation gave suspeciously low results in testing could be randomness but it may be wrong...  

USAGE: 

run:

 python tokenize_data.py --data_path data --save_path token_path --lang 'c++' --test_size 100 --debug_cut_size 0

to make a tokens dataset 

and run:

	python train.py --data_dir token_path/ --save_dir checkpoint/ --epochs 3 --xpu 0

to run training 

