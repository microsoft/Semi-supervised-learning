source activate usb
python train.py --use_tensorboard --gpu 0 --seed 0 --save_name determine \
  --algorithm fixmatch --epoch 10000 --num_train_iter 100000 \
  -ds ag_news -nl 4000 -bsz 32   --accumulate_pseudo_labels True \
  --use_post_hoc_calib False --n_cal 0  --n_th 0  \
  --loss_reweight False --aug_1 weak --aug_2 strong