

def load_dataset(conf):
    data_conf = conf['data_conf']
    dataset_name = data_conf['name']
    
    if(dataset_name == 'general_synthetic'):
        from .synthetic_general import GeneralSynthetic    
        return GeneralSynthetic(data_conf)
    
    elif(dataset_name == 'synth_moon'):
        from .synthetic_moon import SyntheticMoon
        return SyntheticMoon(data_conf)

        
    elif(dataset_name == 'synth_concenteric_circles'):
        from .concentric_circles import ConcentricCircles
        return ConcentricCircles(data_conf)
    
    elif(dataset_name == 'mnist'):
        from .mnist import MNISTData
        return MNISTData(data_conf)

    elif(dataset_name == 'mnist_sklearn'):
        from .mnist_sklearn import MNISTData_sklearn
        return MNISTData_sklearn(data_conf)

    elif(dataset_name == 'cifar10'):
        from .cifar10 import Cifar10Data
        return Cifar10Data(data_conf)
    
    elif(dataset_name == 'cub_birds'):
        from .cub_birds import CUB_BirdsData
        return CUB_BirdsData(data_conf)
    
    elif(dataset_name == 'tiny_imagenet_200'):
        from .tiny_imagenet import TinyImageNet200
        return TinyImageNet200(data_conf)
    
    elif(dataset_name == 'tiny_imagenet_200_CLIP'):
        from .tiny_imagenet_clip import TinyImageNet200CLIP
        return TinyImageNet200CLIP(data_conf)
    
    elif(dataset_name == 'unif_unit_ball'):
        from .uniform_unit_ball import UniformUnitBallDataset
        return UniformUnitBallDataset(data_conf)
    
    elif(dataset_name == 'xor_balls'):
        from .xor_balls import XORBallsDataset
        return XORBallsDataset(data_conf)
    
    elif(dataset_name == 'AG_NEWS'):
        from .text_torch import TextTorch
        return TextTorch(data_conf,dataset_name)
    
    elif(dataset_name == 'IMDB'):
        from .text_torch_emb import TextTorchEmb
        return TextTorchEmb(data_conf,dataset_name)

    elif(dataset_name == 'twenty_newsgroups'):
        from .text_sklearn import TextSklearn
        return TextSklearn(data_conf,dataset_name)
    
    elif(dataset_name == 'multi_nli'):
        from .text_custom import TextCustom
        return TextCustom(data_conf,dataset_name)
    
    elif(dataset_name == 'svhn'):
        from .svhn import SVHNData
        return SVHNData(data_conf)

    elif(dataset_name == 'stl10'):
        from .stl10 import STL10Data
        return STL10Data(data_conf)

    elif(dataset_name == 'stl10_CLIP'):
        from .stl10_clip import STL10CLIP
        return STL10CLIP(data_conf)
    
    elif(dataset_name == 'fashionmnist'):
        from .fashionmnist import FashionMNISTData
        return FashionMNISTData(data_conf)

    else:
        print('Datset {} Not Defined'.format(dataset_name))
        return None

