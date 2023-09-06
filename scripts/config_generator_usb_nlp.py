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


def create_usb_nlp_config(
    alg,
    seed,
    dataset,
    net,
    num_classes,
    num_labels,
    port,
    lr,
    weight_decay,
    layer_decay,
    max_length,
    warmup_epoch=5,
    amp=False,
):
    cfg = {}
    cfg["algorithm"] = alg

    # save config
    cfg["save_dir"] = "./saved_models/usb_nlp"
    cfg["save_name"] = None
    cfg["resume"] = True
    cfg["load_path"] = None
    cfg["overwrite"] = True
    cfg["use_tensorboard"] = True
    cfg["use_wandb"] = False
    cfg["use_aim"] = False

    # algorithm config
    cfg["epoch"] = 100
    cfg["num_train_iter"] = 1024 * 100
    cfg["num_warmup_iter"] = int(1024 * warmup_epoch)
    cfg["num_log_iter"] = 256
    cfg["num_eval_iter"] = 2048
    cfg["num_labels"] = num_labels
    cfg["batch_size"] = 8
    cfg["eval_batch_size"] = 8
    cfg["ema_m"] = 0.0

    if alg == "fixmatch":
        cfg["hard_label"] = True
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
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
    elif alg == "uda":
        cfg["tsa_schedule"] = "exp"
        cfg["T"] = 0.4
        cfg["p_cutoff"] = 0.8
        cfg["ulb_loss_ratio"] = 1.0
    elif alg == "pseudolabel":
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["unsup_warm_up"] = 0.4
    elif alg == "mixmatch":
        cfg["mixup_alpha"] = 0.5
        cfg["T"] = 0.5
        cfg["ulb_loss_ratio"] = 100
        cfg["unsup_warm_up"] = 0.4
        cfg["mixup_manifold"] = True
    elif alg == "remixmatch":
        cfg["mixup_alpha"] = 0.75
        cfg["T"] = 0.5
        cfg["kl_loss_ratio"] = 0.5
        cfg["ulb_loss_ratio"] = 1.5
        cfg["rot_loss_ratio"] = 0.0
        cfg["unsup_warm_up"] = 1 / 64
        cfg["mixup_manifold"] = True
    elif alg == "crmatch":
        cfg["hard_label"] = True
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
        cfg["rot_loss_ratio"] = 0.0
    elif alg == "comatch":
        cfg["hard_label"] = False
        cfg["p_cutoff"] = 0.95
        cfg["contrast_p_cutoff"] = 0.8
        cfg["contrast_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 64
        cfg["queue_batch"] = 128
        cfg["smoothing_alpha"] = 0.9
        cfg["T"] = 0.2
        cfg["da_len"] = 32
    elif alg == "simmatch":
        cfg["p_cutoff"] = 0.95
        cfg["in_loss_ratio"] = 1.0
        cfg["ulb_loss_ratio"] = 1.0
        cfg["proj_size"] = 128
        cfg["K"] = 256
        cfg["da_len"] = 32
        cfg["smoothing_alpha"] = 0.9
        cfg["T"] = 0.1
        cfg["ema_m"] = 0.0
    elif alg == "meanteacher":
        cfg["ulb_loss_ratio"] = 50
        cfg["unsup_warm_up"] = 0.4
    elif alg == "pimodel":
        cfg["ulb_loss_ratio"] = 10
        cfg["unsup_warm_up"] = 0.4
        cfg["ema_m"] = 0.999
    elif alg == "dash":
        cfg["gamma"] = 1.27
        cfg["C"] = 1.0001
        cfg["rho_min"] = 0.05
        cfg["num_wu_iter"] = 2048
        cfg["T"] = 0.5
        cfg["p_cutoff"] = 0.95
        cfg["ulb_loss_ratio"] = 1.0
    elif alg == "mpl":
        cfg["tsa_schedule"] = "exp"
        cfg["T"] = 0.7
        cfg["p_cutoff"] = 0.6
        cfg["ulb_loss_ratio"] = 8.0
        cfg["teacher_lr"] = 0.03
        cfg["label_smoothing"] = 0.1
        cfg["num_uda_warmup_iter"] = 5000
        cfg["num_stu_wait_iter"] = 3000
    elif alg == "vat":
        cfg["vat_embed"] = True
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

    cfg["uratio"] = 1
    cfg["use_cat"] = False

    # optim config
    cfg["optim"] = "AdamW"
    cfg["lr"] = lr
    cfg["momentum"] = 0.9
    cfg["weight_decay"] = weight_decay
    cfg["layer_decay"] = layer_decay
    cfg["amp"] = amp
    cfg["clip"] = 0.0

    # net config
    cfg["net"] = net
    cfg["net_from_name"] = False

    # data config
    cfg["data_dir"] = "./data"
    cfg["dataset"] = dataset
    cfg["train_sampler"] = "RandomSampler"
    cfg["num_classes"] = num_classes
    cfg["num_workers"] = 4
    cfg["max_length"] = max_length

    # basic config
    cfg["seed"] = seed

    # distributed config
    cfg["world_size"] = 1
    cfg["rank"] = 0
    cfg["multiprocessing_distributed"] = False
    cfg["dist_url"] = "tcp://127.0.0.1:" + str(port)
    cfg["dist_backend"] = "nccl"
    cfg["gpu"] = None

    # other config
    cfg["overwrite"] = True

    return cfg


def exp_usb_nlp(label_amount):
    config_file = r"./config/usb_nlp/"
    save_path = r"./saved_models/usb_nlp"

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
        "comatch",
        "crmatch",
        "simmatch",
        "adamatch",
        "softmatch",
        "freematch",
        "defixmatch",
    ]
    datasets = [
        "aclImdb",
        "ag_news",
        "amazon_review",
        "dbpedia",
        "yahoo_answers",
        "yelp_review",
    ]

    seeds = [0]

    dist_port = range(10001, 31120, 1)
    count = 0
    net = "bert_base_uncased"
    weight_decay = 5e-4
    max_length = 512

    for alg in algs:
        for dataset in datasets:
            for seed in seeds:
                # change the configuration of each dataset
                if dataset == "aclImdb":
                    num_classes = 2
                    num_labels = label_amount[0] * num_classes

                    lr = 5e-5
                    layer_decay = 0.75

                elif dataset == "ag_news":
                    num_classes = 4
                    num_labels = label_amount[1] * num_classes

                    lr = 5e-5
                    layer_decay = 0.65

                elif dataset == "amazon_review":
                    num_classes = 5
                    num_labels = label_amount[2] * num_classes

                    lr = 1e-5
                    layer_decay = 0.75

                elif dataset == "dbpedia":
                    num_classes = 14
                    num_labels = label_amount[3] * num_classes
                elif dataset == "yahoo_answers":
                    num_classes = 10
                    num_labels = label_amount[4] * num_classes

                    lr = 1e-4
                    layer_decay = 0.65

                elif dataset == "yelp_review":
                    num_classes = 5
                    num_labels = label_amount[5] * num_classes

                    lr = 5e-5
                    layer_decay = 0.75

                port = dist_port[count]
                # prepare the configuration file
                cfg = create_usb_nlp_config(
                    alg,
                    seed,
                    dataset,
                    net,
                    num_classes,
                    num_labels,
                    port,
                    lr,
                    weight_decay,
                    layer_decay,
                    max_length,
                )
                count += 1
                create_configuration(cfg, config_file)


if __name__ == "__main__":
    if not os.path.exists("./saved_models/usb_nlp/"):
        os.makedirs("./saved_models/usb_nlp/", exist_ok=True)
    if not os.path.exists("./config/usb_nlp/"):
        os.makedirs("./config/usb_nlp/", exist_ok=True)

    # usb nlp
    label_amount = {
        "s": [10, 10, 50, 5, 50, 50],
        "m": [50, 50, 200, 20, 200, 200],
    }

    for i in label_amount:
        exp_usb_nlp(label_amount=label_amount[i])
