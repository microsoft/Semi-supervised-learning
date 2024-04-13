python train.py --use_tensorboard --gpu 0 --seed 0 --save_name fixmatch_plo_e10_i10000_s0_no_acc \
  --algorithm fixmatch --epoch 10 --num_train_iter 10000 \
  -nl 4000 -bsz 32 -ds cifar10 --use_post_hoc_calib True --n_cal 1000  --n_th 1000 --take_d_cal_th_from train_lb