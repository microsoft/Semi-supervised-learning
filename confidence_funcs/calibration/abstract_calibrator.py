from abc import ABC, abstractmethod

class AbstractCalibrator(ABC):
    def __init__(self,clf,val_set,calib_conf):
        '''
        Takes a clf and val_set as input 
        '''
        self.clf = clf 
        self.val_set = val_set 
        self.calib_conf = calib_conf 
    
    @abstractmethod
    def fit(self):
        '''
         fit calibrator
        '''
        pass  
    
    @abstractmethod
    def predict(self):
        '''
          run inference to get uncalibrated outputs
          and then return calibrated output
        '''
        pass
