source activate ssl
python train.py --use_tensorboard --gpu 0 --seed 0 --save_name fixmatch_plo_e10000_i100000_s0_acc_2k_1k_1k_1.0_aug_ur_10 \
  --algorithm fixmatch --epoch 10000 --num_train_iter 100000 \
  -nl 4000 -bsz 32 -ds cifar10 --use_post_hoc_calib True --n_cal 1000  --n_th 1000 --take_d_cal_th_from train_lb --accumulate_pseudo_labels True \
  --ulb_loss_ratio 1.0