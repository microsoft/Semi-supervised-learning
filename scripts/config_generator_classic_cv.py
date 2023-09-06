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


def create_classic_config(
    alg, seed, dataset, net, num_classes, num_labels, img_size, port, weight_decay
):
    cfg = {}
    cfg["algorithm"] = alg

    # save config
    cfg["save_dir"] = "./saved_models/classic_cv"
    cfg["save_name"] = None
    cfg["resume"] = False
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = True
    cfg["use_wandb"] = True
    cfg["use_aim"] = False

    # algorithm config
    cfg["epoch"] = 1024
    cfg["num_train_iter"] = 2**20
    cfg["num_eval_iter"] = 5120
    cfg["num_log_iter"] = 256
    cfg["num_labels"] = num_labels
    cfg["batch_size"] = 64
    cfg["eval_batch_size"] = 256
    if alg == "fixmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["uratio"] = 7
    elif alg == "adamatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["ema_p"] = 0.999
        cfg["uratio"] = 7
    elif alg == "flexmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["thresh_warmup"] = True
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["uratio"] = 7
    elif alg == "uda":
        cfg["tsa_schedule"] = "none"
        cfg["T"] = 0.4
        cfg["p_cutoff"] = 0.8
        cfg["ulb_loss_ratio"] = 1.0
        cfg["uratio"] = 7
    elif alg == "pseudolabel":
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["uratio"] = 1
        cfg["unsup_warm_up"] = 0.4
    elif alg == "mixmatch":
        cfg["uratio"] = 1
        cfg["mixup_alpha"] = 0.5
        cfg["T"] = 0.5
        if dataset == "cifar10":
            cfg["ulb_loss_ratio"] = 100
        elif dataset == "cifar100":
            cfg["ulb_loss_ratio"] = 150
        else:
            cfg["ulb_loss_ratio"] = 100
        cfg["unsup_warm_up"] = 0.4  # 16000 / 1024 / 1024
    elif alg == "remixmatch":
        cfg["mixup_alpha"] = 0.75
        cfg["T"] = 0.5
        cfg["kl_loss_ratio"] = 0.5
        cfg["ulb_loss_ratio"] = 1.5
        cfg["rot_loss_ratio"] = 0.5
        cfg["unsup_warm_up"] = 1 / 64
        cfg["uratio"] = 1
    elif alg == "crmatch":
        cfg["hard_label"] = True
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["uratio"] = 7
    elif alg == "comatch":
        cfg["hard_label"] = False
        cfg["p_cutoff"] = 0.95
        cfg["contrast_p_cutoff"] = 0.8
        cfg["contrast_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 64
        cfg["queue_batch"] = 5
        cfg["smoothing_alpha"] = 0.9
        cfg["uratio"] = 7
        cfg["T"] = 0.2
        cfg["da_len"] = 32

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

    elif alg == "simmatch":
        cfg["p_cutoff"] = 0.95
        cfg["in_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 128
        cfg["K"] = 256
        cfg["da_len"] = 32
        cfg["smoothing_alpha"] = 0.9
        cfg["uratio"] = 7
        if dataset in ["cifar10", "svhn", "cifar100", "stl10"]:
            cfg["T"] = 0.1
        else:
            cfg["T"] = 0.2
    elif alg == "meanteacher":
        cfg["uratio"] = 1
        cfg["ulb_loss_ratio"] = 50
        cfg["unsup_warm_up"] = 0.4
    elif alg == "pimodel":
        cfg["ulb_loss_ratio"] = 10
        cfg["uratio"] = 1
        cfg["unsup_warm_up"] = 0.4
    elif alg == "dash":
        cfg["gamma"] = 1.27
        cfg["C"] = 1.0001
        cfg["rho_min"] = 0.05
        cfg["num_wu_iter"] = 2048
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["uratio"] = 7
    elif alg == "mpl":
        cfg["tsa_schedule"] = "none"
        cfg["T"] = 0.7
        cfg["p_cutoff"] = 0.6
        cfg["ulb_loss_ratio"] = 8.0
        cfg["uratio"] = 7
        cfg["teacher_lr"] = 0.03
        cfg["label_smoothing"] = 0.1
        cfg["num_uda_warmup_iter"] = 5000
        cfg["num_stu_wait_iter"] = 3000

    elif alg == "freematch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["ema_p"] = 0.999
        cfg["ent_loss_ratio"] = 0.001
        cfg["uratio"] = 7
        cfg["use_quantile"] = False
        if dataset == "svhn":
            cfg["clip_thresh"] = True
    elif alg == "softmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["dist_align"] = True
        cfg["dist_uniform"] = True
        cfg["per_class"] = False
        cfg["ema_p"] = 0.999
        cfg["ulb_loss_ratio"] = 1.0
        cfg["n_sigma"] = 2
        cfg["uratio"] = 7
        if dataset == "imagenet":
            cfg["ulb_loss_ratio"] = 1.0
    elif alg == "defixmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 0.5
        cfg["uratio"] = 7

    # cfg['img']
    cfg["ema_m"] = 0.999
    cfg["crop_ratio"] = 0.875
    cfg["img_size"] = img_size

    # optim config
    cfg["optim"] = "SGD"
    cfg["lr"] = 0.03
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["layer_decay"] = 1.0
    cfg["amp"] = False
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
    cfg["num_workers"] = 1

    # basic config
    cfg["seed"] = seed

    # distributed config
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = True
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"
    cfg["gpu"] = None

    # other config
    cfg["overwrite"] = True
    cfg["amp"] = False
    cfg["use_wandb"] = False
    cfg["use_aim"] = False

    return cfg


# prepare the configuration for baseline model, use_penalty == False
def exp_classic_cv(label_amount):
    config_file = r"./config/classic_cv/"
    save_path = r"./saved_models/classic_cv"

    if not os.path.exists(config_file):
        os.makedirs(config_file)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
    datasets = ["cifar100", "svhn", "stl10", "cifar10"]

    seeds = [0]

    dist_port = range(10001, 11120, 1)
    count = 0

    for alg in algs:
        for dataset in datasets:
            for seed in seeds:
                # change the configuration of each dataset
                if dataset == "cifar10":
                    # net = 'WideResNet'
                    num_classes = 10
                    num_labels = label_amount[0]
                    weight_decay = 5e-4
                    net = "wrn_28_2"
                    img_size = 32

                elif dataset == "cifar100":
                    # net = 'WideResNet'
                    num_classes = 100
                    num_labels = label_amount[1]
                    weight_decay = 1e-3
                    # depth = 28
                    # widen_factor = 8
                    # net = 'wrn_28_8'
                    net = "wrn_28_2"
                    img_size = 32

                elif dataset == "svhn":
                    # net = 'WideResNet'
                    num_classes = 10
                    num_labels = label_amount[2]
                    weight_decay = 5e-4
                    # depth = 28
                    # widen_factor = 2
                    net = "wrn_28_2"
                    img_size = 32

                elif dataset == "stl10":
                    # net = 'WideResNetVar'
                    num_classes = 10
                    num_labels = label_amount[3]
                    weight_decay = 5e-4
                    net = "wrn_var_37_2"
                    img_size = 96

                elif dataset == "imagenet":
                    if alg not in ["fixmatch", "flexmatch"]:
                        continue
                    net = "resnet50"
                    num_classes = 1000
                    num_labels = 100000  # 128000
                    weight_decay = 3e-4

                port = dist_port[count]
                # prepare the configuration file
                cfg = create_classic_config(
                    alg,
                    seed,
                    dataset,
                    net,
                    num_classes,
                    num_labels,
                    img_size,
                    port,
                    weight_decay,
                )
                count += 1
                create_configuration(cfg, config_file)


if __name__ == "__main__":
    # if not os.path.exists('./saved_models/classic_cv/'):
    #     os.mkdir('./saved_models/classic_cv/')
    if not os.path.exists("./config/classic_cv/"):
        os.mkdir("./config/classic_cv/")

    # classic cv
    label_amount = {
        "s": [40, 400, 40, 40],
        "m": [250, 2500, 250, 250],
        "l": [4000, 10000, 1000, 1000],
    }

    for i in label_amount:
        exp_classic_cv(label_amount=label_amount[i])
