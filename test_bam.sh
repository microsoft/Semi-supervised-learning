export OPT_TH=YES
conda activate usb
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml --use_post_hoc_calib False --n_cal 3000 --n_th 3000 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 --accumulate_pseudo_labels True --bayes False
python train.py --c config/classic_cv/fixmatch/fixmatch_cifar10_250_0.yaml --use_post_hoc_calib False --n_cal 3000 --n_th 3000 --take_d_cal_th_from eval --loss_reweight False --aug_1 weak --aug_2 strong --gpu 0 --accumulate_pseudo_labels True --bayes True --bam_config ./config/bam/default.yaml
