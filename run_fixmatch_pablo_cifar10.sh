source activate ssl
export OPT_TH="YES"
python train.py --use_tensorboard --gpu 1 --seed 0 --prefix cb_eval --save_name determine \
  --algorithm fixmatch --epoch 1000 --num_train_iter 50000 --num_classes 10 \
  -ds cifar10 --use_cat False -nl 250 -bsz 64 --eval_batch_size 256 --accumulate_pseudo_labels True \
  --use_post_hoc_calib True --n_cal 1000  --n_th 1000 --take_d_cal_th_from eval \
  --aug_1 weak --aug_2 strong \
  --full_pl_flag True --batch_pl_flag False --full_pl_freq 100 \
  --c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml 