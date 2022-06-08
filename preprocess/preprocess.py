
import os
torchssl_dir = os.path.dirname(os.path.dirname(__file__))
import sys 
sys.path.append(torchssl_dir)
from torchssl.datasets.ssl_dataset import SSL_Dataset

def generate_idx(dataset, num_classes, num_labels, seed):

    class Args:
        algorithm = 'fixmatch'
        dataset = dataset 
        num_classes = num_classes
        num_labels = num_labels 
        seed = seed 
    
    args = Args()

    train_dset = SSL_Dataset(args, alg=args.algorithm, name=args.dataset, train=True,
                            num_classes=args.num_classes, data_dir='./data')
    lb_dset, ulb_dset = train_dset.get_ssl_dset(args.num_labels)


if __name__ == '__main__':

    seed_list = [0, 1, 2]
    data_info = {
        'cifar10': (10, [40, 250, 4000]),
        'cifar100': (100, [400, 2500, 10000]),
        'stl10': (10, [40, 250, 1000]),
        'svhn': (10, [40, 250, 1000])
    }

    for dataset, (num_classes, num_labels_list) in data_info.items():
        for num_labels in num_labels_list:
            for seed in seed_list:
                print(f"dataset: {dataset}, num_labels: {num_labels}, seed: {seed}")
                generate_idx(dataset, num_classes, num_labels, seed)
