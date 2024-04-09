

def get_model(model_conf, logger):
    
    logger.info(model_conf)
    model_name = model_conf['name']

    if(model_name=='binary_logistic_regression'):
        from .logistic_regression import  PyTorchLogisticRegression
        assert model_conf['num_classes']==2 
        model= PyTorchLogisticRegression(model_conf,logger)
    
    elif(model_name=='lenet'):
        from .lenet import LeNet5
        model = LeNet5(model_conf['num_classes'])
    
    elif(model_name=='linear_model'):
        from .linear_model import LinearModel
        model = LinearModel(model_conf)
    
    elif(model_name=='cifar_small_net'):
        from .cifar_small_net import CifarSmallNet
        model = CifarSmallNet(model_conf['num_classes'])
    
    elif(model_name=='resnet18'):
        from .resnet import ResNet18
        model = ResNet18(model_conf['num_classes'])
    
    elif(model_name=='resnet18_v2'):
        from .resnet_v2 import resnet18_v2
        model =  resnet18_v2(n_classes=model_conf['num_classes'])

    elif(model_name=='cifar_med_net'):
        from .cifar_medium_net import CifarMediumNet
        model = CifarMediumNet(model_conf['num_classes'])
    
    elif(model_name=='temp_scaling'):
        from .scaling_model import TemperatureScalingModel
        model = TemperatureScalingModel(model_conf)
    
    elif(model_name=='two_layer_net'):
        from .two_layer_nets import TwoLayerNet
        model = TwoLayerNet(model_conf)
    
    elif(model_name == "mlp"):
        from .mlp import MLP
        model = MLP(model_conf)
        
    elif(model_name=='ViT'):
        from .cifar_vit_small import ViT
        model = ViT(model_conf)
    
    elif(model_name== 'TextClassificationModel'): 
        from .text_embed_mlp import TextClassificationModel
        model = TextClassificationModel(model_conf) 

    elif(model_name == 'text_clf_mlp_head'):
        from .text_clf_mlp_head import TextClassifierMLPHead
        model = TextClassifierMLPHead(model_conf)

    elif(model_name == 'g_model_head'):
        from .mlp_head import MLPHead
        model = MLPHead(model_conf,bb_model)

    elif(model_name == 'dynamic_mlp'):
        from .dynamic_mlp import DynamicMLP 
        model = DynamicMLP(model_conf)
    
    elif(model_name == 'simplenet'):
        from .simplenet import SimpleNet
        model = SimpleNet(num_classes=model_conf['num_classes'])
    
    return model 