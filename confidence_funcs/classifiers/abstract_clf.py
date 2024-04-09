from abc import ABC, abstractmethod

class AbstractClassifier(ABC):
    
    @abstractmethod
    def fit(self,dataset,training_conf):
        pass
    
    @abstractmethod
    def predict(self, dataset, inference_conf):
        pass
    
    @abstractmethod
    def get_weights(self):
        pass

