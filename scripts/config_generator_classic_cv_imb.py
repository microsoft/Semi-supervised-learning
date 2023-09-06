# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Create the .yaml for each experiment
"""
import os

from config_generator_classic_cv import create_classic_config
from config_generator_usb_cv import create_usb_cv_config


def create_configuration(cfg, cfg_file):
    if "imb_algorithm" in cfg:
        cfg[
            "save_name"
        ] = "{alg}_{imb_alg}_{dataset}_lb{num_lb}_{imb_lb}_ulb{num_ulb}_{imb_ulb}_{seed}".format(  # noqa: E501
            imb_alg=cfg["imb_algorithm"],
            alg=cfg["algorithm"],
            dataset=cfg["dataset"],
            num_lb=cfg["num_labels"],
            imb_lb=cfg["lb_imb_ratio"],
            num_ulb=cfg["ulb_num_labels"],
            imb_ulb=cfg["ulb_imb_ratio"],
            seed=cfg["seed"],
        )
    else:
        cfg[
            "save_name"
        ] = "{alg}_{dataset}_lb{num_lb}_{imb_lb}_ulb{num_ulb}_{imb_ulb}_{seed}".format(
            alg=cfg["algorithm"],
            dataset=cfg["dataset"],
            num_lb=cfg["num_labels"],
            imb_lb=cfg["lb_imb_ratio"],
            num_ulb=cfg["ulb_num_labels"],
            imb_ulb=cfg["ulb_imb_ratio"],
            seed=cfg["seed"],
        )

    # resume
    cfg["resume"] = True
    cfg["load_path"] = "{}/{}/latest_model.pth".format(
        cfg["save_dir"], cfg["save_name"]
    )

    if "imb_algorithm" in cfg:
        alg_file = cfg_file + cfg["algorithm"] + "_" + cfg["imb_algorithm"] + "/"
    else:
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


def create_classic_cv_imb_config(
    alg,
    seed,
    dataset,
    net,
    num_classes,
    num_labels,
    img_size,
    port,
    lr,
    weight_decay,
    imb_alg,
    lb_imb_ratio,
    ulb_imb_ratio,
    ulb_num_labels,
    amp=False,
    layer_decay=1.0,
    pretrain_name=None,
):
    # get core algorithm related configs
    if dataset in ["cifar10", "cifar100", "stl10"]:
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
        cfg["net"] = net
        cfg["optim"] = "SGD"
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay
        cfg["num_workers"] = 1
        cfg["epoch"] = 256
        cfg["num_train_iter"] = cfg["epoch"] * 1024
        cfg["num_eval_iter"] = 1024
        cfg["lr"] = lr
    elif dataset in ["semi_aves", "semi_aves_out"]:
        cfg = create_usb_cv_config(
            alg,
            seed,
            dataset,
            net,
            num_classes,
            num_labels,
            img_size,
            0.875,
            port,
            lr,
            weight_decay,
            layer_decay,
            pretrain_name,
            warmup=0,
            amp=False,
        )

        cfg["optim"] = "AdamW"
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay
        cfg["num_workers"] = 4
        cfg["epoch"] = 128
        cfg["num_train_iter"] = cfg["epoch"] * 1024
        cfg["num_eval_iter"] = 1024
    elif dataset in ["imagenet", "imagenet127"]:
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
        cfg["net"] = net
        cfg["optim"] = "SGD"
        cfg["lr"] = lr
        cfg["weight_decay"] = weight_decay
        cfg["num_workers"] = 4
        cfg["epoch"] = 200
        cfg["num_train_iter"] = cfg["epoch"] * 2500
        cfg["num_eval_iter"] = 2500
        cfg["batch_size"] = 256
        cfg["img_size"] = img_size
        cfg["crop_ratio"] = 0.875

    cfg["num_log_iter"] = 256

    cfg["include_lb_to_ulb"] = False
    cfg["use_cat"] = True
    cfg["net"] = net
    cfg["lb_imb_ratio"] = lb_imb_ratio
    cfg["ulb_imb_ratio"] = ulb_imb_ratio
    cfg["ulb_num_labels"] = ulb_num_labels

    # rewrite configs for imbalanced setting
    if imb_alg is not None:
        cfg["imb_algorithm"] = imb_alg

    cfg["save_dir"] = "./saved_models/classic_cv_imb"
    cfg["data_dir"] = "./data"

    if alg == "fixmatch":
        cfg["uratio"] = 2
    elif alg == "remixmatch":
        cfg["uratio"] = 1
        cfg["dist_align_target"] = "uniform"

    if imb_alg == "crest+":
        cfg["epoch"] = 128
        cfg["num_train_iter"] = 2**16
        cfg["crest_num_gens"] = 6
        cfg["crest_pro_dist_align"] = True
        cfg["crest_alpha"] = 3
        cfg["crest_dist_align_t"] = 0.5
        if dataset in ["imagenet", "imagenet127"]:
            cfg["epoch"] = 100
            cfg["num_train_iter"] = cfg["epoch"] * 2500
            cfg["crest_num_gens"] = 3
            cfg["crest_alpha"] = 0.7
            cfg["crest_dist_align_t"] = 0.5
    elif imb_alg == "crest":
        cfg["epoch"] = 128
        cfg["num_train_iter"] = 2**16
        cfg["crest_num_gens"] = 6
        cfg["crest_pro_dist_align"] = False
        cfg["crest_alpha"] = 3
        if dataset in ["imagenet", "imagenet127"]:
            cfg["epoch"] = 100
            cfg["num_train_iter"] = cfg["epoch"] * 2500
            cfg["crest_num_gens"] = 3
            cfg["crest_alpha"] = 0.7
            cfg["crest_dist_align_t"] = 0.5
    elif imb_alg == "darp":
        # cfg['epoch'] = 512
        # cfg['num_train_iter'] = 2 ** 18
        cfg["darp_warmup_epochs"] = 200
        if dataset in ["imagenet", "imagenet127"]:
            cfg["darp_warmup_epochs"] = 150
        cfg["darp_alpha"] = 2.0
        cfg["darp_num_refine_iter"] = 10
        cfg["darp_iter_T"] = 10
    elif imb_alg == "abc":
        # cfg['epoch'] = 512
        # cfg['num_train_iter'] = 2 ** 18
        cfg["abc_p_cutoff"] = 0.95
        cfg["abc_loss_ratio"] = 1.0
    elif imb_alg == "daso":
        cfg["daso_queue_len"] = 256
        cfg["daso_T_proto"] = 0.05
        cfg["daso_interp_alpha"] = 0.5
        cfg["daso_with_dist_aware"] = True
        cfg["daso_assign_loss_ratio"] = 1.0
        cfg["daso_num_pl_dist_iter"] = 100
        cfg["daso_num_pretrain_iter"] = 5120
        if dataset == "cifar10":
            cfg["daso_T_dist"] = 1.5
        else:
            cfg["daso_T_dist"] = 0.3
    elif imb_alg == "cossl":
        cfg["cossl_max_lam"] = 0.6
        cfg["cossl_tfe_augment"] = "strong"
        cfg["cossl_tfe_u_ratio"] = 1
        cfg["cossl_warm_epoch"] = 200
        if dataset in ["imagenet127"]:
            cfg["cossl_warm_epoch"] = 150
        cfg["cossl_tfe_warm_epoch"] = 10
        cfg["cossl_tfe_warm_lr"] = 0.02
        cfg["cossl_tfe_warm_ema_decay"] = 0.999
        cfg["cossl_tfe_warm_wd"] = 5e-4
        cfg["cossl_tfe_warm_bs"] = 64
    elif imb_alg == "tras":
        cfg["tras_A"] = 2
        cfg["tras_B"] = 2
        cfg["tras_tro"] = 1.0
        cfg["tras_warmup_epochs"] = 1
    else:
        pass

    cfg["use_wandb"] = False
    cfg["use_aim"] = False
    return cfg


def exp_classic_cv_imb(settings):
    config_file = r"./config/classic_cv_imb/"
    save_path = r"./saved_models/classic_cv_imb"

    if not os.path.exists(config_file):
        os.mkdir(config_file)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    algs = ["supervised", "fixmatch", "remixmatch"]
    imb_algs = [
        None,
        "crest",
        "crest+",
        "darp",
        "abc",
        "daso",
        "saw",
        "adsh",
        "cossl",
        "debiaspl",
        "simis",
    ]
    datasets = list(settings.keys())

    seeds = [0]

    dist_port = range(10001, 31120, 1)
    count = 0
    layer_decay = 1.0

    for alg in algs:
        for imb_alg in imb_algs:
            if alg == "remixmatch" and imb_alg in ["adsh", "tars"]:
                continue

            for dataset in datasets:
                for seed in seeds:
                    if dataset == "cifar10":
                        num_classes = 10
                        img_size = 32
                        net = "wrn_28_2"
                        lr = 0.03
                        weight_decay = 5e-4
                    elif dataset == "cifar100":
                        num_classes = 100
                        img_size = 32
                        net = "wrn_28_2"
                        lr = 0.03
                        weight_decay = 5e-4
                    elif dataset == "stl10":
                        num_classes = 10
                        img_size = 32
                        net = "wrn_28_2"
                        lr = 0.03
                        weight_decay = 5e-4
                    elif dataset == "semi_aves" or dataset == "semi_aves_out":
                        num_classes = 200

                        img_size = 224

                        net = "vit_small_patch16_224"
                        pretrain_name = "https://github.com/microsoft/Semi-supervised-learning/releases/download/v.0.0.0/vit_small_patch16_224_mlp_im_1k_224.pth"  # noqa: E501

                        lr = 1e-3
                        layer_decay = 0.65
                    elif dataset == "imagenet127":
                        num_classes = 127
                        img_size = 112
                        lr = 0.1
                        weight_decay = 1e-4
                        net = "resnet50"

                    if dataset in ["cifar10", "cifar100"]:
                        setting_dicts = settings[dataset]

                        for setting_dict in setting_dicts:
                            lb_imb_ratio = setting_dict["lb_imb_ratio"]
                            ulb_imb_ratio = setting_dict["ulb_imb_ratio"]
                            lb_num_labels = setting_dict["lb_num_labels"]
                            ulb_num_labels = setting_dict["ulb_num_labels"]

                            if alg == "supervised":
                                port = dist_port[count]
                                # prepare the configuration file
                                cfg = create_classic_cv_imb_config(
                                    alg,
                                    seed,
                                    dataset,
                                    net,
                                    num_classes,
                                    lb_num_labels,
                                    img_size,
                                    port,
                                    lr,
                                    weight_decay,
                                    None,
                                    lb_imb_ratio,
                                    ulb_imb_ratio,
                                    ulb_num_labels,
                                    amp=False,
                                )
                                count += 1
                                create_configuration(cfg, config_file)
                            else:
                                port = dist_port[count]
                                # prepare the configuration file
                                cfg = create_classic_cv_imb_config(
                                    alg,
                                    seed,
                                    dataset,
                                    net,
                                    num_classes,
                                    lb_num_labels,
                                    img_size,
                                    port,
                                    lr,
                                    weight_decay,
                                    imb_alg,
                                    lb_imb_ratio,
                                    ulb_imb_ratio,
                                    ulb_num_labels,
                                    amp=False,
                                )
                                count += 1
                                create_configuration(cfg, config_file)

                    elif dataset == "stl10":
                        setting_dicts = settings[dataset]

                        for setting_dict in setting_dicts:
                            lb_imb_ratio = setting_dict["lb_imb_ratio"]
                            ulb_imb_ratio = 1.0
                            lb_num_labels = setting_dict["lb_num_labels"]
                            ulb_num_labels = None

                            if alg == "supervised":
                                port = dist_port[count]
                                # prepare the configuration file
                                cfg = create_classic_cv_imb_config(
                                    alg,
                                    seed,
                                    dataset,
                                    net,
                                    num_classes,
                                    lb_num_labels,
                                    img_size,
                                    port,
                                    lr,
                                    weight_decay,
                                    None,
                                    lb_imb_ratio,
                                    ulb_imb_ratio,
                                    ulb_num_labels,
                                    amp=False,
                                )
                                count += 1
                                create_configuration(cfg, config_file)
                            else:
                                port = dist_port[count]
                                # prepare the configuration file
                                cfg = create_classic_cv_imb_config(
                                    alg,
                                    seed,
                                    dataset,
                                    net,
                                    num_classes,
                                    lb_num_labels,
                                    img_size,
                                    port,
                                    lr,
                                    weight_decay,
                                    imb_alg,
                                    lb_imb_ratio,
                                    ulb_imb_ratio,
                                    ulb_num_labels,
                                    amp=False,
                                )
                                count += 1
                                create_configuration(cfg, config_file)

                    elif dataset in ["semi_aves", "semi_aves_out"]:
                        lb_num_labels = 3957
                        lb_imb_ratio = 1.0
                        ulb_num_labels = None
                        ulb_imb_ratio = 1.0

                        if alg == "supervised":
                            port = dist_port[count]
                            # prepare the configuration file
                            cfg = create_classic_cv_imb_config(
                                alg,
                                seed,
                                dataset,
                                net,
                                num_classes,
                                lb_num_labels,
                                img_size,
                                port,
                                lr,
                                weight_decay,
                                None,
                                lb_imb_ratio,
                                ulb_imb_ratio,
                                ulb_num_labels,
                                amp=False,
                                layer_decay=layer_decay,
                                pretrain_name=pretrain_name,
                            )
                            count += 1
                            create_configuration(cfg, config_file)
                        else:
                            port = dist_port[count]
                            # prepare the configuration file
                            cfg = create_classic_cv_imb_config(
                                alg,
                                seed,
                                dataset,
                                net,
                                num_classes,
                                lb_num_labels,
                                img_size,
                                port,
                                lr,
                                weight_decay,
                                imb_alg,
                                lb_imb_ratio,
                                ulb_imb_ratio,
                                ulb_num_labels,
                                amp=False,
                                layer_decay=layer_decay,
                                pretrain_name=pretrain_name,
                            )
                            count += 1
                            create_configuration(cfg, config_file)

                    elif dataset in ["imagenet127"]:
                        lb_num_labels = 128101
                        lb_imb_ratio = 286
                        ulb_num_labels = None
                        ulb_imb_ratio = 286
                        img_size = 112
                        seed = 0

                        if alg == "supervised":
                            port = dist_port[count]
                            # prepare the configuration file
                            cfg = create_classic_cv_imb_config(
                                alg,
                                seed,
                                dataset,
                                net,
                                num_classes,
                                lb_num_labels,
                                img_size,
                                port,
                                lr,
                                weight_decay,
                                None,
                                lb_imb_ratio,
                                ulb_imb_ratio,
                                ulb_num_labels,
                                amp=False,
                            )
                            count += 1
                            create_configuration(cfg, config_file)
                        else:
                            port = dist_port[count]
                            # prepare the configuration file
                            cfg = create_classic_cv_imb_config(
                                alg,
                                seed,
                                dataset,
                                net,
                                num_classes,
                                lb_num_labels,
                                img_size,
                                port,
                                lr,
                                weight_decay,
                                imb_alg,
                                lb_imb_ratio,
                                ulb_imb_ratio,
                                ulb_num_labels,
                                amp=False,
                            )
                            count += 1
                            create_configuration(cfg, config_file)


if __name__ == "__main__":
    if not os.path.exists("./saved_models/classic_cv_imb/"):
        os.makedirs("./saved_models/classic_cv_imb/", exist_ok=True)
    if not os.path.exists("./config/classic_cv_imb/"):
        os.makedirs("./config/classic_cv_imb/", exist_ok=True)

    settings = {
        "cifar10": [
            {
                "lb_num_labels": 1500,
                "ulb_num_labels": 3000,
                "lb_imb_ratio": 100,
                "ulb_imb_ratio": 100,
            },
            {
                "lb_num_labels": 500,
                "ulb_num_labels": 4000,
                "lb_imb_ratio": 100,
                "ulb_imb_ratio": 100,
            },
            {
                "lb_num_labels": 1500,
                "ulb_num_labels": 3000,
                "lb_imb_ratio": 150,
                "ulb_imb_ratio": 150,
            },
            {
                "lb_num_labels": 500,
                "ulb_num_labels": 4000,
                "lb_imb_ratio": 150,
                "ulb_imb_ratio": 150,
            },
            {
                "lb_num_labels": 1500,
                "ulb_num_labels": 3000,
                "lb_imb_ratio": 100,
                "ulb_imb_ratio": -100,
            },
            {
                "lb_num_labels": 500,
                "ulb_num_labels": 4000,
                "lb_imb_ratio": 100,
                "ulb_imb_ratio": -100,
            },
        ],
        "cifar100": [
            {
                "lb_num_labels": 150,
                "ulb_num_labels": 300,
                "lb_imb_ratio": 10,
                "ulb_imb_ratio": 10,
            },
            {
                "lb_num_labels": 150,
                "ulb_num_labels": 300,
                "lb_imb_ratio": 15,
                "ulb_imb_ratio": 15,
            },
            {
                "lb_num_labels": 150,
                "ulb_num_labels": 300,
                "lb_imb_ratio": 10,
                "ulb_imb_ratio": -10,
            },
        ],
        "stl10": [
            {"lb_num_labels": 150, "lb_imb_ratio": 10},
            {"lb_num_labels": 150, "lb_imb_ratio": 20},
        ],
        "imagenet127": [],
    }

    exp_classic_cv_imb(settings)
