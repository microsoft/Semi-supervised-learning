# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .fixmatch import FixMatch
from .flexmatch import FlexMatch
from .pimodel import PiModel
from .meanteacher import MeanTeacher
from .pseudolabel import PseudoLabel
from .uda import UDA
from .mixmatch import MixMatch
from .vat import VAT
from .remixmatch import ReMixMatch
from .crmatch import CRMatch
from .dash import Dash
# from .mpl import MPL
from .fullysupervised import FullySupervised
from .comatch import CoMatch
from .simmatch import SimMatch
from .adamatch import AdaMatch
# add new algorithms here

from semilearn.core.utils import ALGORITHMS
name2alg = ALGORITHMS


def get_algorithm(args, net_builder, tb_log, logger):
    if args.algorithm in ALGORITHMS:
        alg = ALGORITHMS[args.algorithm]( # name2alg[args.algorithm](
            args=args,
            net_builder=net_builder,
            tb_log=tb_log,
            logger=logger
        )
        return alg
    else:
        raise KeyError(f'Unknown algorithm: {str(args.algorithm)}')



