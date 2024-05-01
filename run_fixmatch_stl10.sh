python train.py --c config/classic_cv/fixmatch/fixmatch_stl10_1000_0.yaml --use_post_hoc_calib False --n_cal 2400 --n_th 2400 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 1> no_calib.out 2> no_calib.err &
python train.py --c config/classic_cv/fixmatch/fixmatch_stl10_1000_0.yaml --use_post_hoc_calib True --n_cal 2400 --n_th 2400 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 1 1> calib_no_acc.out 2> calib_no_acc.err &
python train.py --c config/classic_cv/fixmatch/fixmatch_stl10_1000_0.yaml --use_post_hoc_calib True --n_cal 2400 --n_th 2400 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 --accumulate_pseudo_labels True 1> calib_yes_acc.out 2> calib_yes_acc.err &