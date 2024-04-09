# In the name of God the most compassionate the most merciful
""" SimpleNet

Paper: `Lets Keep it simple, Using simple architectures to outperform deeper and more complex architectures`
    - https://arxiv.org/abs/1608.06037

@article{hasanpour2016lets,
  title={Lets keep it simple, Using simple architectures to outperform deeper and more complex architectures},
  author={Hasanpour, Seyyed Hossein and Rouhani, Mohammad and Fayyaz, Mohsen and Sabokrou, Mohammad},
  journal={arXiv preprint arXiv:1608.06037},
  year={2016}
}

Official Caffe impl at https://github.com/Coderx7/SimpleNet
Official Pythorch impl at https://github.com/Coderx7/SimpleNet_Pytorch
Seyyed Hossein Hasanpour
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file

from typing import Union, Tuple, List, Dict, Any, cast, Optional

# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = "center"
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


__all__ = [
    "simplenet",
    "simplenet_cifar_310k",
    "simplenet_cifar_460k",
    "simplenet_cifar_5m",
    "simplenet_cifar_5m_extra_pool",  # refers to the early pytorch impl of cifar10/100 that mistakenly used an extra pooling(see issue #5)
    "simplenetv1_small_m1_05",  # 1.5m
    "simplenetv1_small_m2_05",  # 1.5m
    "simplenetv1_small_m1_075",  # 3m
    "simplenetv1_small_m2_075",  # 3m
    "simplenetv1_5m_m1",  # 5m
    "simplenetv1_5m_m2",  # 5m
    "simplenetv1_9m_m1",  # 9m
    "simplenetv1_9m_m2",  # 9m
]  # model_registry will add each entrypoint fn to this


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "crop_pct": 0.875,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        **kwargs,
    }


default_cfgs: Dict[str, Dict[str, Any]] = {
    "simplenetv1_small_m1_05": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m1_05-be804903.pth"
    ),
    "simplenetv1_small_m2_05": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m2_05-ca4b3e2b.pth"
    ),
    "simplenetv1_small_m1_075": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m1_075-098acbff.pth"
    ),
    "simplenetv1_small_m2_075": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_small_m2_075-609ff4da.pth"
    ),
    "simplenetv1_5m_m1": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_5m_m1-36c4ca4d.pth"
    ),
    "simplenetv1_5m_m2": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_5m_m2-9bd6bb36.pth"
    ),
    "simplenetv1_9m_m1": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_9m_m1-524f9972.pth"
    ),
    "simplenetv1_9m_m2": _cfg(
        url="https://github.com/Coderx7/SimpleNet_Pytorch/releases/download/v1.0.0/simplenetv1_9m_m2-59e8733b.pth"
    ),
}


class View(nn.Module):
    def forward(self, x):
        print(f"{x.shape}")
        return x


class SimpleNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        in_chans: int = 3,
        scale: float = 1,
        network_idx: int = 2,
        mode: int = 0,
        drop_rates: Dict[int, float] = {},
    ):
        """Instantiates a SimpleNet model. SimpleNet is comprised of the most basic building blocks of a CNN architecture.
        It uses basic principles to maximize the network performance both in terms of feature representation and speed without
        resorting to complex design or operators. 
        
        Args:
            num_classes (int, optional): number of classes. Defaults to 1000.
            in_chans (int, optional): number of input channels. Defaults to 3.
            scale (float, optional): scale of the architecture width. Defaults to 1.0.
            network_idx (int, optional): the network index indicating the 5 million or 8 million version(0 and 1 respectively). Defaults to 0.
            mode (int, optional): stride mode of the architecture. specifies how fast the input shrinks.
                This is used for larger input sizes such as the 224x224 in imagenet training where the
                input size incurs a lot of overhead if not downsampled properly. 
                you can choose between 0 meaning no change and 4. where each number denotes a specific
                downsampling strategy. For imagenet use 1-4.
                the larger the stride mode, the higher accuracy and the slower
                the network gets. stride mode 1 is the fastest and achives very good accuracy.
                Defaults to 2.
            drop_rates (Dict[int,float], optional): custom drop out rates specified per layer. 
                each rate should be paired with the corrosponding layer index(pooling and cnn layers are counted only). Defaults to {}.
        """
        super(SimpleNet, self).__init__()
        # (channels or layer-type, stride=1, drp=0.)
        self.cfg: Dict[str, List[Tuple[Union(int, str), int, Union(float, None), Optional[str]]]] = {
            "simplenet_cifar_310k": [
                (64, 1, 0.0),
                (32, 1, 0.0),
                (32, 1, 0.0),
                (32, 1, None),
                ("p", 2, 0.0),
                (32, 1, 0.0),
                (32, 1, 0.0),
                (64, 1, None),
                ("p", 2, 0.0),
                (64, 1, 0.0),
                (64, 1, None),
                ("p", 2, 0.0),
                (128, 1, 0.0),
                (256, 1, 0.0, "k1"),
                (64, 1, None, "k1"),
                ("p", 2, 0.0),
                (64, 1, None),
            ],
            "simplenet_cifar_460k": [
                (32, 1, 0.0),
                (32, 1, 0.0),
                (32, 1, 0.0),
                (64, 1, None),
                ("p", 2, 0.0),
                (64, 1, 0.0),
                (64, 1, 0.0),
                (64, 1, None),
                ("p", 2, 0.0),
                (64, 1, 0.0),
                (64, 1, None),
                ("p", 2, 0.0),
                (96, 1, 0.0),
                (96, 1, 0.0, "k1"),
                (96, 1, None, "k1"),
                ("p", 2, 0.0),
                (100, 1, None),
            ],
            "simplenet_cifar_5m": [
                (64, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, None),
                ("p", 2, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                (256, 1, None),
                ("p", 2, 0.0),
                (256, 1, 0.0),
                (256, 1, None),
                ("p", 2, 0.0),
                (512, 1, 0.0),
                (2048, 1, 0.0, "k1"),
                (256, 1, None, "k1"),
                ("p", 2, 0.0),
                (256, 1, None),
            ],
            "simplenet_cifar_5m_extra_pool": [
                (64, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, None),
                ("p", 2, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                (256, 1, None),
                ("p", 2, 0.0),
                (256, 1, 0.0),
                (256, 1, None),
                ("p", 2, 0.0),
                (512, 1, 0.0),
                ("p", 2, 0.0),  # extra pooling!
                (2048, 1, 0.0, "k1"),
                (256, 1, None, "k1"),
                ("p", 2, 0.0),
                (256, 1, None),
            ],
            "simplenetv1_imagenet": [
                (64, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                (128, 1, 0.0),
                ("p", 2, 0.0),
                (256, 1, 0.0),
                (256, 1, 0.0),
                (256, 1, 0.0),
                (512, 1, 0.0),
                ("p", 2, 0.0),
                (2048, 1, 0.0, "k1"),
                (256, 1, 0.0, "k1"),
                (256, 1, 0.0),
            ],
            "simplenetv1_imagenet_9m": [
                (128, 1, 0.0),
                (192, 1, 0.0),
                (192, 1, 0.0),
                (192, 1, 0.0),
                (192, 1, 0.0),
                (192, 1, 0.0),
                ("p", 2, 0.0),
                (320, 1, 0.0),
                (320, 1, 0.0),
                (320, 1, 0.0),
                (640, 1, 0.0),
                ("p", 2, 0.0),
                (2560, 1, 0.0, "k1"),
                (320, 1, 0.0, "k1"),
                (320, 1, 0.0),
            ],
        }
        # make sure values are in proper form!
        self.dropout_rates = {int(key): float(value) for key, value in drop_rates.items()}
        # 15 is the last layer of the network(including two previous pooling layers)
        # basically specifying the dropout rate for the very last layer to be used after the pooling
        self.last_dropout_rate = self.dropout_rates.get(15, 0.0)
        self.strides = {
            0: {},
            1: {0: 2, 1: 2, 2: 2},
            2: {0: 2, 1: 2, 2: 1, 3: 2},
            3: {0: 2, 1: 2, 2: 1, 3: 1, 4: 2},
            4: {0: 2, 1: 1, 2: 2, 3: 1, 4: 2, 5: 1},
        }

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.scale = scale
        self.networks = [
            "simplenetv1_imagenet",  # 0
            "simplenetv1_imagenet_9m",  # 1
            # other archs
            "simplenet_cifar_310k",  # 2
            "simplenet_cifar_460k",  # 3
            "simplenet_cifar_5m",  # 4
            "simplenet_cifar_5m_extra_pool",  # 5
        ]
        self.network_idx = network_idx
        self.mode = mode

        self.features = self._make_layers(scale)
        self.classifier = nn.Linear(round(self.cfg[self.networks[network_idx]][-1][0] * scale), num_classes)

    def forward(self, x):
        out = self.features(x)
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        out = F.dropout2d(out, self.last_dropout_rate, training=self.training)
        out = out.view(out.size(0), -1)

        outputs = {}


        outputs['pre_logits'] = out
        logits = self.classifier(out)

        outputs['features'] = logits
        outputs['logits'] = logits
        outputs['abs_logits'] =  torch.abs(logits)

        outputs['probs'] = F.softmax(logits)

        return outputs

    def _make_layers(self, scale: float):
        layers: List[nn.Module] = []
        input_channel = self.in_chans
        stride_list = self.strides[self.mode]
        for idx, (layer, stride, defaul_dropout_rate, *layer_type) in enumerate(
            self.cfg[self.networks[self.network_idx]]
        ):
            stride = stride_list[idx] if len(stride_list) > idx else stride
            # check if any custom dropout rate is specified
            # for this layer, note that pooling also counts as 1 layer
            custom_dropout = self.dropout_rates.get(idx, None)
            custom_dropout = defaul_dropout_rate if custom_dropout is None else custom_dropout
            # dropout values must be strictly decimal. while 0 doesnt introduce any issues here
            # i.e. during training and inference, if you try to jit trace your model it will crash
            # due to using 0 as dropout value(this applies up to 1.13.1) so here is an explicit
            # check to convert any possible integer value to its decimal counterpart.
            custom_dropout = None if custom_dropout is None else float(custom_dropout)
            kernel_size = 3
            padding = 1
            if layer_type == ['k1']:
                kernel_size = 1
                padding = 0 

            if layer == "p":
                layers += [
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(stride, stride)),
                    nn.Dropout2d(p=custom_dropout, inplace=True),
                ]
            else:
                filters = round(layer * scale)
                if custom_dropout is None:
                    layers += [
                        nn.Conv2d(input_channel, filters, kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [
                        nn.Conv2d(input_channel, filters, kernel_size=kernel_size, stride=stride, padding=padding),
                        nn.BatchNorm2d(filters, eps=1e-05, momentum=0.05, affine=True),
                        nn.ReLU(inplace=True),
                        nn.Dropout2d(p=custom_dropout, inplace=False),
                    ]

                input_channel = filters

        model = nn.Sequential(*layers)
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        return model


def _gen_simplenet(
    model_variant: str = "simplenetv1_m2",
    num_classes: int = 1000,
    in_chans: int = 3,
    scale: float = 1.0,
    network_idx: int = 0,
    mode: int = 2,
    pretrained: bool = False,
    drop_rates: Dict[int, float] = {},
) -> SimpleNet:
    model = SimpleNet(num_classes, in_chans, scale=scale, network_idx=network_idx, mode=mode, drop_rates=drop_rates)
    if pretrained:
        cfg = default_cfgs.get(model_variant, None)
        if cfg is None:
            raise Exception(f"Unknown model variant ('{model_variant}') specified!")
        url = cfg["url"]
        checkpoint_filename = url.split("/")[-1]
        checkpoint_path = f"tmp/{checkpoint_filename}"
        print(f"saving in checkpoint_path:{checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            os.makedirs("tmp", exist_ok=True)
            download_url_to_file(url, checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu",)
        model.load_state_dict(checkpoint)
    return model


def simplenet(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Generic simplenet model builder. by default it returns `simplenetv1_5m_m2` model
    but specifying different arguments such as `netidx`, `scale` or `mode` will result in 
    the corrosponding network variant. 
    
    when pretrained is specified, if the combination of settings resemble any known variants
    specified in the `default_cfg`, their respective pretrained weights will be loaded, otherwise
    an exception will be thrown denoting Unknown model variant being specified.  

    Args:
        pretrained (bool, optional): loads the model with pretrained weights only if the model is a known variant specified in default_cfg. Defaults to False.

    Raises:
        Exception: if pretrained is used with an unknown/custom model variant and exception is raised.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    num_classes = kwargs.get("num_classes", 1000)
    in_chans = kwargs.get("in_chans", 3)
    scale = kwargs.get("scale", 1.0)
    network_idx = kwargs.get("network_idx", 0)
    mode = kwargs.get("mode", 2)
    drop_rates = kwargs.get("drop_rates", {})
    model_variant = "simplenetv1_5m_m2"
    if pretrained:
        # check if the model specified is a known variant
        model_base = None
        if network_idx == 0:
            model_base = 5
        elif network_idx == 1:
            model_base = 9
        config = ""
        if math.isclose(scale, 1.0):
            config = f"{model_base}m_m{mode}"
        elif math.isclose(scale, 0.75):
            config = f"small_m{mode}_075"
        elif math.isclose(scale, 0.5):
            config = f"small_m{mode}_05"
        else:
            config = f"m{mode}_{scale:.2f}".replace(".", "")
        model_variant = f"simplenetv1_{config}"

    return _gen_simplenet(model_variant, num_classes, in_chans, scale, network_idx, mode, pretrained, drop_rates)


def _remove_network_settings(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Removes network related settings passed in kwargs for predefined network configruations below

    Returns:
        Dict[str,Any]: cleaned kwargs
    """
    model_args = {k: v for k, v in kwargs.items() if k not in ["scale", "network_idx", "mode", "drop_rate"]}
    return model_args


# cifar10/100 models
def simplenet_cifar_310k(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """original implementation of smaller variants of simplenet for cifar10/100 
    that were used in the paper 
    """
    model_variant = "simplenet_cifar_310k"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, network_idx=2, mode=0, pretrained=pretrained, **model_args)


def simplenet_cifar_460k(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """original implementation of smaller variants of simplenet for cifar10/100 
    that were used in the paper 
    """
    model_variant = "simplenet_cifar_460k"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, network_idx=3, mode=0, pretrained=pretrained, **model_args)


def simplenet_cifar_5m(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """The original implementation of simplenet trained on cifar10/100 in caffe.
    """
    model_variant = "simplenet_cifar_5m"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, network_idx=4, mode=0, pretrained=pretrained, **model_args)


def simplenet_cifar_5m_extra_pool(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """An early pytorch implementation of simplenet that mistakenly used an extra pooling layer
    .it was not know until 2021 which was reported in https://github.com/Coderx7/SimpleNet_Pytorch/issues/5
    this is just here to be able to load the weights that were trained using this variation still available on the repository. 
    """
    model_variant = "simplenet_cifar_5m_extra_pool"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, network_idx=5, mode=0, pretrained=pretrained, **model_args)


# imagenet models
def simplenetv1_small_m1_05(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a small variant of simplenetv1_5m, with 1.5m parameters. This uses m1 stride mode
    which makes it the fastest variant available. 
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_small_m1_05"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.5, network_idx=0, mode=1, pretrained=pretrained, **model_args)


def simplenetv1_small_m2_05(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a second small variant of simplenetv1_5m, with 1.5m parameters. This uses m2 stride mode
    which makes it the second fastest variant available.  
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_small_m2_05"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.5, network_idx=0, mode=2, pretrained=pretrained, **model_args)


def simplenetv1_small_m1_075(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a third small variant of simplenetv1_5m, with 3m parameters. This uses m1 stride mode
    which makes it the third fastest variant available.  
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_small_m1_075"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.75, network_idx=0, mode=1, pretrained=pretrained, **model_args)


def simplenetv1_small_m2_075(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a forth small variant of simplenetv1_5m, with 3m parameters. This uses m2 stride mode
    which makes it the forth fastest variant available. 
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_small_m2_075"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=0.75, network_idx=0, mode=2, pretrained=pretrained, **model_args)


def simplenetv1_5m_m1(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates the base simplement model known as simplenetv1_5m, with 5m parameters. This variant uses m1 stride mode
    which makes it a fast and performant model.  
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_5m_m1"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=0, mode=1, pretrained=pretrained, **model_args)


def simplenetv1_5m_m2(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates the base simplement model known as simplenetv1_5m, with 5m parameters. This variant uses m2 stride mode
    which makes it a bit more performant model compared to the m1 variant of the same variant at the expense of a bit slower inference.  
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_5m_m2"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=0, mode=2, pretrained=pretrained, **model_args)


def simplenetv1_9m_m1(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a variant of the simplenetv1_5m, with 9m parameters. This variant uses m1 stride mode
    which makes it run faster.  
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_9m_m1"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=1, mode=1, pretrained=pretrained, **model_args)


def simplenetv1_9m_m2(pretrained: bool = False, **kwargs: Any) -> SimpleNet:
    """Creates a variant of the simplenetv1_5m, with 9m parameters. This variant uses m2 stride mode
    which makes it a bit more performant model compared to the m1 variant of the same variant at the expense of a bit slower inference.  
    
    Args:
        pretrained (bool, optional): loads the model with pretrained weights. Defaults to False.

    Returns:
        SimpleNet: a SimpleNet model instance is returned upon successful instantiation. 
    """
    model_variant = "simplenetv1_9m_m2"
    model_args = _remove_network_settings(kwargs)
    return _gen_simplenet(model_variant, scale=1.0, network_idx=1, mode=2, pretrained=pretrained, **model_args)


if __name__ == "__main__":
    model = simplenet(network_idx=4, mode=0, num_classes=10)
    input_dummy = torch.randn(size=(1, 3, 32, 32))
    out = model(input_dummy)
    # out.mean().backward()
    print(model)
    print(f"output: {out.size()}")
