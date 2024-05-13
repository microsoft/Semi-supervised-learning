source activate ssl
OPT_TH="YES"

python train.py --use_tensorboard --gpu 0 --seed 0 --prefix cb_eval --save_name determine \
  --algorithm fixmatch --epoch 1000 --num_train_iter 50000 --num_classes 10 \
  -ds cifar10 --use_cat False -nl 250 -bsz 64 --eval_batch_size 256 --accumulate_pseudo_labels False \
  --use_post_hoc_calib False --n_cal 0  --n_th 0 --take_d_cal_th_from eval \
  --aug_1 weak --aug_2 strong \
  --c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml 