# CUDA_VISIBLE_DEVICES=6 python transformer.py ../results_3/movie/dm.token/model.json ../results_3/movie/dm.token/path.json --is_train
CUDA_VISIBLE_DEVICES=4 python transformer.py ../results_3/movie/dm.token/model.json ../results_3/movie/dm.token/path.json --eval_set tst --resume_file ../results_3/movie/dm.token/model/epoch.44.th