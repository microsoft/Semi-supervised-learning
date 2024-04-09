import sys 
sys.path.append('../../')
from .abstract_clf import AbstractClassifier
import numpy as np
from math import exp 
from sklearn.metrics import accuracy_score

class CustomLogisticRegression(AbstractClassifier):

    def __init__(self,model_conf,logger=None):
        self.input_dim = model_conf['input_dimension']
        self.num_classes = model_conf['num_classes']
        self.model_conf = model_conf
        self.margin_function = 'dot_product'
        self.fit_intercept = model_conf['fit_intercept']

        #self.logger = logger
    
    def set_defaults(self,training_conf):
        training_conf.setdefault('loss_tol',1e-5)
        training_conf.setdefault('max_epochs',500)
        training_conf.setdefault('verbose',False)
        training_conf.setdefault('learning_rate',1.0)
        training_conf.setdefault('normalize_weights',False)
    
    def fit(self,train_dataset,training_conf):
        self.set_defaults(training_conf)

        X,Y = train_dataset.X, train_dataset.Y
        n,d = X.shape
        
        if(self.fit_intercept):
            X_ = np.zeros((n,d+1))
            X_[:,:d] = X
            X_[:,-1] = 1
            d = d+1
            X = X_ 

        
        w =  np.zeros(d) 
        #w[0] = 100
        #w = np.array([ -0.70814278, -0.70606926, 1.44712748])
        
        #w[0]= 
        lr = training_conf['learning_rate']
        
        ## Y in the input are assumed to be in {0,1}
        ## convert to {-1,1}
        Y_ = 2*Y -1 
        max_epochs = training_conf['max_epochs']

        for e in range(max_epochs):
            s = np.zeros(n)
            for i in range(n):
                s[i] = 1+ exp(-Y_[i]*np.dot(w,X[i]))
                
            nll_loss = np.log(s).sum()/n
            g = np.sum([(Y_[i]*X[i]*(1-s[i]))/(s[i]) for i in range(n)],axis=0)/n
            w = w - lr*g 

            # Projection step, it hurts the convergence of logistic regression
            # Disable it and project in the end. 
            # why does projection hurt convergence.
            # it seems to prefer high norm w to get 0 training error.
            #w = w/np.linalg.norm(w)
            
            if(self.fit_intercept):
                self.w = w[:d-1]
                self.b = w[-1]
            else:
                self.w = w 
                self.b = 0

            te = self.test_error(train_dataset)
            if(te <= 1e-4):
                print('Training error',te)
                break 

            if(e%50==0 and training_conf['verbose']):
                lr = lr*0.99
                print(w)
                print(nll_loss,lr,np.linalg.norm(g),te)
            if(e%1000==0 and te>1e-2):
                #w = np.random.uniform(size=d)
                w =  np.zeros(d) 
                w[0] = 10
                lr = 1e-1

        if(training_conf['normalize_weights']):
            c = np.linalg.norm(w)
            self.w = self.w/c
            self.b = self.b/c

        return self.w
        
    def predict(self,dataset,inference_conf=None,):
        w = self.w 
        b = self.b 
        X = dataset.X
        
        n,d = X.shape
        Y_hat =  np.sign(np.dot(X,w) + b )
        
        P_hat = np.array([1/(1+np.exp(-Y_hat[i]*(np.dot(X[i],w)+b))) for i in range(n)] )
        
        Y_hat = ((Y_hat+1)/2).astype(int)

        if(self.margin_function=='dot_product'):
            margin_score = np.array([ float(abs(np.dot(self.w,x) + b)) for x in X])
        else:
            margin_score = P_hat 
        
        out = {}
        out['labels']     = Y_hat 
        out['confidence'] = P_hat 
        out['margin_score'] = margin_score
        return out 

    def test_error(self,dataset):
        Y = dataset.Y
        inf_out  = self.predict(dataset)
        return 1 - accuracy_score(inf_out['labels'],Y)

    def get_weights(self):
        return self.w 
    
