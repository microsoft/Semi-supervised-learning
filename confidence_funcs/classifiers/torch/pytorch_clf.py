
from ..abstract_clf import AbstractClassifier
from .models import model_factory
from .model_training import *
from .clf_inference import *
from .clf_get_embedding import *

class PyTorchClassifier(AbstractClassifier):

    def __init__(self,model_conf,logger,bb_model=None):

        self.model = model_factory.get_model(model_conf, logger)
        self.model_conf = model_conf 
        # if model_conf['should_compile']:
        #     self.model = torch.compile(self.model)
        self.logger = logger 
        
            
    def fit(self,dataset,training_conf,val_set=None):
        method = training_conf['method'] if "method" in training_conf else None         
        if(method==None or method=="vanila"):
            model_training = ModelTraining(self.logger)
            out = model_training.train(self.model,dataset,training_conf,val_set=val_set)
        return out 


    def predict(self, dataset, inference_conf=None):
        clf_inference = ClassfierInference(self.logger)
        return clf_inference.predict(self.model,dataset,inference_conf)

    def get_weights(self):
        w = torch.nn.utils.parameters_to_vector(self.model.parameters()).detach().cpu()
        return w
    
    def scale_weights(self,c):
        for p in self.model.parameters():
            p.data *= c

    def norm(self):
        total_norm = 0
        for p in self.model.parameters():
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm

    def get_grad_embedding(self,dataset, inference_conf):
        clf_embedding = ClassifierEmbedding(self.logger)
        return clf_embedding.get_gard_embedding(self.model,dataset,inference_conf)

    def get_embedding(self,dataset, inference_conf):
        clf_embedding = ClassifierEmbedding(self.logger)
        return clf_embedding.get_embedding(self.model,dataset,inference_conf)
    
    def grad_norm(self):
        model = self.model
        for p in model.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm