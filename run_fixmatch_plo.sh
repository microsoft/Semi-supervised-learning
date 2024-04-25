python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0.yaml --accumulate_pseudo_labels True \
  --use_post_hoc_calib True --n_cal 500  --n_th 500 --take_d_cal_th_from eval \
  --loss_reweight False --aug_1 weak --aug_2 strong