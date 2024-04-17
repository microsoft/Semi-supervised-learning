source activate ssl
python train.py --use_tensorboard --seed 0 --save_name fixmatch-e100_i100000_s0_4k_acc --algorithm fixmatch --epoch 100 --num_train_iter 100000\
  -nl 4000 -bsz 32 -ds cifar10 --use_post_hoc_calib False  --n_cal 0  --n_th 0 --take_d_cal_th_from train_lb --accumulate_pseudo_labels True