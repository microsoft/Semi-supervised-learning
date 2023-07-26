# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# will be released soon
from semilearn.algorithms import name2alg
from semilearn.core.utils import IMB_ALGORITHMS
name2imbalg = IMB_ALGORITHMS


def get_imb_algorithm(args, net_builder, tb_log, logger):
    if args.imb_algorithm not in name2imbalg:
        print(f'Unknown imbalanced algorithm: {args.imb_algorithm }')

    class DummyClass(name2imbalg[args.imb_algorithm], name2alg[args.algorithm]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
    alg = DummyClass(args=args, net_builder=net_builder, tb_log=tb_log, logger=logger)
    return alg