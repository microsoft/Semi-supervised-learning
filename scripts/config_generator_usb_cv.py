# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Create the .yaml for each experiment
"""
import os


def create_configuration(cfg, cfg_file):
    cfg["save_name"] = "{alg}_{dataset}_{num_lb}_{seed}".format(
        alg=cfg["algorithm"],
        dataset=cfg["dataset"],
        num_lb=cfg["num_labels"],
        seed=cfg["seed"],
    )

    # resume
    cfg["resume"] = True
    cfg["load_path"] = "{}/{}/latest_model.pth".format(
        cfg["save_dir"], cfg["save_name"]
    )

    alg_file = cfg_file + cfg["algorithm"] + "/"
    if not os.path.exists(alg_file):
        os.mkdir(alg_file)

    print(alg_file + cfg["save_name"] + ".yaml")
    with open(alg_file + cfg["save_name"] + ".yaml", "w", encoding="utf-8") as w:
        lines = []
        for k, v in cfg.items():
            line = str(k) + ": " + str(v)
            lines.append(line)
        for line in lines:
            w.writelines(line)
            w.write("\n")


def create_usb_cv_config(
    alg,
    seed,
    dataset,
    net,
    num_classes,
    num_labels,
    img_size,
    crop_ratio,
    port,
    lr,
    weight_decay,
    layer_decay,
    pretrain_path,
    warmup=5,
    amp=False,
):
    cfg = {}
    cfg["algorithm"] = alg

    # save config
    cfg["save_dir"] = "./saved_models/usb_cv/"
    cfg["save_name"] = None
    cfg["resume"] = False
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = True
    cfg["use_wandb"] = False
    cfg["use_aim"] = False

    if dataset == "imagenet":
        cfg["epoch"] = 500
        cfg["num_train_iter"] = 1024 * 500
        cfg["num_log_iter"] = 256
        cfg["num_eval_iter"] = 5120
        cfg["batch_size"] = 256
        cfg["eval_batch_size"] = 512
    else:
        cfg["epoch"] = 200
        cfg["num_train_iter"] = 1024 * 200
        cfg["num_log_iter"] = 256
        cfg["num_eval_iter"] = 2048
        cfg["batch_size"] = 8
        cfg["eval_batch_size"] = 16

    cfg["num_warmup_iter"] = int(1024 * warmup)
    cfg["num_labels"] = num_labels

    cfg["uratio"] = 1
    cfg["ema_m"] = 0.0

    if alg == "fixmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 10.0
            cfg["p_cutoff"] = 0.7
    elif alg == "adamatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["ema_p"] = 0.999
    elif alg == "flexmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["thresh_warmup"] = True
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 10.0
            cfg["p_cutoff"] = 0.7
    elif alg == "uda":
        cfg["tsa_schedule"] = "none"
        cfg["T"] = 0.4
        cfg["p_cutoff"] = 0.8
        cfg["ulb_loss_ratio"] = 1.0
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 10.0
    elif alg == "pseudolabel":
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["unsup_warm_up"] = 0.4
    elif alg == "mixmatch":
        cfg["mixup_alpha"] = 0.5
        cfg["T"] = 0.5
        cfg["ulb_loss_ratio"] = 10
        cfg["unsup_warm_up"] = 0.4  # 16000 / 1024 / 1024
    elif alg == "remixmatch":
        cfg["mixup_alpha"] = 0.75
        cfg["T"] = 0.5
        cfg["kl_loss_ratio"] = 0.5
        cfg["ulb_loss_ratio"] = 1.5
        cfg["rot_loss_ratio"] = 0.5
        cfg["unsup_warm_up"] = 1 / 64
    elif alg == "crmatch":
        cfg["hard_label"] = True
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0

    elif alg == "comatch":
        cfg["hard_label"] = False
        cfg["p_cutoff"] = 0.95
        cfg["contrast_p_cutoff"] = 0.8
        cfg["contrast_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 64
        cfg["queue_batch"] = 32
        cfg["smoothing_alpha"] = 0.9
        cfg["T"] = 0.2
        cfg["da_len"] = 32
        cfg["ema_m"] = 0.999
        if dataset == "stl10":
            cfg["contrast_loss_ratio"] = 5.0

        if dataset == "imagenet":
            cfg["p_cutoff"] = 0.6
            cfg["contrast_p_cutoff"] = 0.3
            cfg["contrast_loss_ratio"] = 10.0
            cfg["ulb_loss_ratio"] = 10.0
            cfg["smoothing_alpha"] = 0.9
            cfg["T"] = 0.1
            cfg["proj_size"] = 128
            cfg["queue_batch"] = 128

    elif alg == "simmatch":
        cfg["p_cutoff"] = 0.95
        cfg["in_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 128
        cfg["K"] = 256
        cfg["da_len"] = 32
        cfg["smoothing_alpha"] = 0.9
        cfg["ema_m"] = 0.999

        if dataset in ["cifar10", "svhn", "cifar100", "stl10"]:
            cfg["T"] = 0.1
        else:
            cfg["T"] = 0.2

        if dataset == "imagenet":
            cfg["in_loss_ratio"] = 5.0
            cfg["ulb_loss_ratio"] = 10.0
            cfg["T"] = 0.1
            cfg["p_cutoff"] = 0.7
            cfg["da_len"] = 256
            cfg["ema_m"] = 0.999

    elif alg == "meanteacher":
        cfg["ulb_loss_ratio"] = 50
        cfg["unsup_warm_up"] = 0.4
        cfg["ema_m"] = 0.999

    elif alg == "pimodel":
        cfg["ulb_loss_ratio"] = 10

        cfg["unsup_warm_up"] = 0.4
    elif alg == "dash":
        cfg["gamma"] = 1.27
        cfg["C"] = 1.0001
        cfg["rho_min"] = 0.05
        cfg["num_wu_iter"] = 2048
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0

    elif alg == "mpl":
        cfg["tsa_schedule"] = "none"
        cfg["T"] = 0.7
        cfg["p_cutoff"] = 0.6
        cfg["ulb_loss_ratio"] = 8.0

        cfg["teacher_lr"] = 0.03
        cfg["label_smoothing"] = 0.1
        cfg["num_uda_warmup_iter"] = 5000
        cfg["num_stu_wait_iter"] = 3000

    elif alg == "freematch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["ema_p"] = 0.999
        cfg["ent_loss_ratio"] = 0.001
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 1.0
    elif alg == "softmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["dist_align"] = True
        cfg["dist_uniform"] = True
        cfg["per_class"] = False
        cfg["ema_p"] = 0.999
        cfg["ulb_loss_ratio"] = 1.0
        cfg["n_sigma"] = 2
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 1.0
    elif alg == "defixmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 0.5

    cfg["img_size"] = img_size
    cfg["crop_ratio"] = crop_ratio

    # optim config
    cfg["optim"] = "AdamW"
    cfg["lr"] = lr
    cfg["layer_decay"] = layer_decay
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["amp"] = amp
    cfg["clip"] = 0.0
    cfg["use_cat"] = True

    # net config
    cfg["net"] = net
    cfg["net_from_name"] = False

    # data config
    cfg["data_dir"] = "./data"
    cfg["dataset"] = dataset
    cfg["train_sampler"] = "RandomSampler"
    cfg["num_classes"] = num_classes
    cfg["num_workers"] = 4

    # basic config
    cfg["seed"] = seed

    # distributed config
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = True
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"
    cfg["gpu"] = None

    if alg == "crmatch" and dataset == "stl10":
        cfg["multiprocessing_distributed"] = False
        cfg["gpu"] = 0

    # other config
    cfg["overwrite"] = True
    cfg["use_pretrain"] = True
    cfg["pretrain_path"] = pretrain_path

    return cfg


# prepare the configuration for baseline model, use_penalty == False
def exp_usb_cv(label_amount):
    config_file = r"./config/usb_cv/"
    save_path = r"./saved_models/usb_cv"

    if not os.path.exists(config_file):
        os.mkdir(config_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    algs = [
        "flexmatch",
        "fixmatch",
        "uda",
        "pseudolabel",
        "fullysupervised",
        "supervised",
        "remixmatch",
        "mixmatch",
        "meanteacher",
        "pimodel",
        "vat",
        "dash",
        "crmatch",
        "comatch",
        "simmatch",
        "adamatch",
        "freematch",
        "softmatch",
        "defixmatch",
    ]
    datasets = ["cifar100", "eurosat", "semi_aves", "tissuemnist", "stl10"]
    # algs = ['fixmatch', 'flexmatch', 'comatch', 'simmatch']
    # datasets = ['imagenet']
    # seeds = [0, 1, 2]  # 1, 22, 333
    seeds = [0]

    dist_port = range(10001, 11120, 1)
    count = 0

    pretrain_path = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0"  # noqa: E501
    weight_decay = 5e-4
    # lr = 5e-5
    warmup = 0
    amp = False

    for alg in algs:
        for dataset in datasets:
            for seed in seeds:
                # change the configuration of each dataset
                if dataset == "cifar10":
                    num_classes = 10
                    num_labels = label_amount[0] * num_classes

                    img_size = 32
                    crop_ratio = 0.875
                    net = "vit_tiny_patch2_32"
                    pretrain_name = "vit_tiny_patch2_32_mlp_im_1k_32.pth"

                    lr = 5e-4
                    layer_decay = 0.5

                elif dataset == "cifar100":
                    num_classes = 100
                    num_labels = label_amount[1] * num_classes

                    # depth = 28
                    # widen_factor = 8
                    img_size = 32
                    crop_ratio = 0.875
                    net = "vit_small_patch2_32"
                    pretrain_name = "vit_small_patch2_32_mlp_im_1k_32.pth"

                    lr = 5e-4
                    layer_decay = 0.5

                elif dataset == "svhn":
                    img_size = 32
                    crop_ratio = 0.875

                    num_classes = 10
                    num_labels = label_amount[2] * num_classes

                    net = "vit_tiny_patch2_32"
                    pretrain_name = "vit_tiny_patch2_32_mlp_im_1k_32.pth"

                elif dataset == "stl10":
                    num_classes = 10
                    num_labels = label_amount[3] * num_classes
                    img_size = 96
                    crop_ratio = 0.875
                    warmup = 5

                    net = "vit_base_patch16_96"
                    pretrain_name = "mae_pretrain_vit_base.pth"

                    lr = 1e-4
                    layer_decay = 0.65

                elif dataset == "semi_aves":
                    num_classes = 200
                    # num_labels = label_amount[4] * num_classes
                    num_labels = 3959

                    img_size = 224
                    crop_ratio = 0.875

                    net = "vit_small_patch16_224"
                    pretrain_name = "vit_small_patch16_224_mlp_im_1k_224.pth"

                    lr = 1e-3
                    layer_decay = 0.65

                # NOTE: resize to 32 x 32
                elif dataset == "eurosat":
                    num_classes = 10
                    num_labels = label_amount[5] * num_classes

                    img_size = 32
                    crop_ratio = 0.875

                    net = "vit_small_patch2_32"
                    pretrain_name = "vit_small_patch2_32_mlp_im_1k_32.pth"

                    lr = 5e-5
                    layer_decay = 1.0

                elif dataset == "tissuemnist":
                    num_classes = 8
                    num_labels = label_amount[6] * num_classes
                    img_size = 32
                    crop_ratio = 0.95

                    net = "vit_tiny_patch2_32"
                    pretrain_name = "vit_tiny_patch2_32_mlp_im_1k_32.pth"

                    lr = 5e-5
                    layer_decay = 0.95

                elif dataset == "imagenet":
                    net = "vit_base_path16_224"
                    pretrain_name = "mae_pretrain_vit_base.pth"
                    num_classes = 1000
                    num_labels = 100000  # 128000
                    lr = 1e-3
                    weight_decay = 0.01
                    warmup = 5
                    amp = True
                    img_size = 224
                    crop_ratio = 0.875

                port = dist_port[count]
                # prepare the configuration file
                cfg = create_usb_cv_config(
                    alg,
                    seed,
                    dataset,
                    net,
                    num_classes,
                    num_labels,
                    img_size,
                    crop_ratio,
                    port,
                    lr,
                    weight_decay,
                    layer_decay,
                    pretrain_path=os.path.join(pretrain_path, pretrain_name),
                    warmup=warmup,
                    amp=amp,
                )
                count += 1
                create_configuration(cfg, config_file)


if __name__ == "__main__":
    if not os.path.exists("./saved_models/usb_cv/"):
        os.makedirs("./saved_models/usb_cv/", exist_ok=True)
    if not os.path.exists("./config/usb_cv/"):
        os.makedirs("./config/usb_cv/", exist_ok=True)

    label_amount = {"s": [2, 2, 2, 4, 2, 2, 10], "m": [4, 4, 4, 10, 2, 4, 50]}

    for i in label_amount:
        exp_usb_cv(label_amount=label_amount[i])
