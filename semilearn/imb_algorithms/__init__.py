from .crest import CReST
from semilearn.algorithms import name2alg


name2imbalg = {
    'crest': CReST,
}


def get_imb_algorithm(args, net_builder, tb_log, logger):
    class DummyClass(name2imbalg[args.imb_algorithm], name2alg[args.algorithm]):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    try:
        alg = DummyClass(args=args, net_builder=net_builder, tb_log=tb_log, logger=logger)
        return alg
    except KeyError as e:
        print(f'Unknown imbalanced algorithm: {str(e)}')