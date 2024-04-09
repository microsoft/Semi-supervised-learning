from sklearn.linear_model import LogisticRegression
from ..abstract_clf import AbstractClassifier
import numpy as np
import sys 
from sklearn import svm 
sys.path.append('../../')
from scipy.special import softmax
#from math import abs

from sklearn.svm import SVC,NuSVC


class SkLearnClassifier(AbstractClassifier):

    def __init__(self,model_conf,logger):
        self.input_dim = model_conf['input_dimension']
        self.num_classes = model_conf['num_classes']
        self.model_conf = model_conf
        self.logger = logger
        
        

    def create_model(self,seed=0):
        model_conf = self.model_conf
        training_conf = self.training_conf
        #assert self.num_classes == 2
        assert self.input_dim   >= 1
        t_conf = training_conf
        
        self.logger.debug('model_conf : {}'.format(model_conf))

        if(model_conf['model_name']=='logistic_regression'):
            self.model = LogisticRegression(fit_intercept=model_conf['fit_intercept'],
                                            max_iter =  t_conf['max_epochs'],
                                            tol = t_conf['loss_tolerance'],
                                            solver= t_conf['optimizer_name'],
                                            penalty=t_conf['regularization'],
                                            C=t_conf['C']
                                            )
            
        if(model_conf['model_name']=='svm'): 
            #self.model = svm.LinearSVC(loss='hinge',tol=t_conf['loss_tolerance'],C=t_conf['C'],
            #                            max_iter=t_conf['max_epochs'],fit_intercept=model_conf['fit_intercept'])
            kernel ='linear'
            if('kernel' in model_conf):
                kernel = model_conf['kernel']
            
            if model_conf['fit_intercept'] == False:
                self.logger.debug('Training Linear SVC')
                self.model = svm.LinearSVC(loss='squared_hinge',tol=t_conf['loss_tolerance'],C=t_conf['C'],
                                        max_iter=t_conf['max_epochs'],fit_intercept=model_conf['fit_intercept'],
                                        class_weight="balanced",random_state=seed,dual=False)
            else:

                #print(f'seed here {seed}')
                self.model = SVC(kernel=kernel,random_state=seed,C=t_conf['C'], 
                                    tol = t_conf['loss_tolerance'],
                                    probability=t_conf['probability'])


        self.margin_function =  'dot_product'
        self.fit_intercept = model_conf['fit_intercept']

        
    def set_default_conf(self,training_conf) :
        
        training_conf.setdefault('C',1)
        training_conf.setdefault('optimizer_name','lbfgs')
        #training_conf.setdefault('learning_rate',1e-2)
        training_conf.setdefault('loss_tol',1e-5)
        training_conf.setdefault('max_epochs',1000)
        training_conf.setdefault('normalize_weights',False)
        training_conf.setdefault('seed',20000)
        training_conf.setdefault('probability',False)

        training_conf.setdefault('regularization','l2')
        

    
    def fit(self,train_dataset,training_conf,val_set=None):
        
        self.normalize_weights = training_conf['normalize_weights']

        self.set_default_conf(training_conf)

        self.training_conf = training_conf


        self.create_model(seed=training_conf['seed'])

        X = train_dataset.X 
        Y = train_dataset.Y 

        self.logger.info(f'Training Data Size : {len(X)}')

        n,d = X.shape
        assert d == self.input_dim
        assert Y.min() < self.num_classes and Y.min() > -1


        self.model.fit(X,Y)
        #print(self.model.coef_.shape)

        if(self.num_classes == 2):
            self.w = self.model.coef_[0]
        else:
            self.w = self.model.coef_
        
        self.b = self.model.intercept_
        
        #if(self.b> np.max(X)):  
        #    self.b = (1/10)*np.max(X)

        if(self.normalize_weights):
            w = self.model.coef_ [0]
            z = np.linalg.norm(w)
            self.w = w/z
            #self.model.coef_ = w 
            if(self.fit_intercept):
                self.b = self.model.intercept_ /z 
                #self.model.intercept_ = self.model.intercept_ /z 

    
    def predict(self,test_dataset,inference_conf=None):
        #X = dataset.test_dataset.X

        X = test_dataset.X
        #Y_hat = self.model.predict(X)
        #P_hat = self.model.predict_proba(X)
        
        #print(self.w,self.b)
        out = {}
        #print(self.w)
        scores = np.array([np.dot(self.w,x) + self.b for x in X]) #.squeeze()

        if self.training_conf['probability']:
            out['probs'] = self.model.predict_proba(X) #softmax(scores,axis=1)#np.exp(scores)/(np.sum(np.exp(scores)),axis)
        else:
            
            Z = np.zeros((len(scores),2))
            Z[:,0] = np.array(1/(1+np.exp(scores))).reshape(len(scores),)
            Z[:,1] = np.array(1-(1/(1+np.exp(scores)))).reshape(len(scores),)
            #a = 1/(1+np.exp(1))
            #b = 1/(1+np.exp(-1))
            #Z = (Z -a)/(b-a)
            out['probs'] = Z
            #out['probs'] = self.model.predict_proba(X)

        #if self.num_classes > 2: # convert from one-hot to actual labels
            #Y_hat = np.argmax(scores,axis=1)
        Y_hat = self.model.predict(X)
        #else:
            #Y_hat = ((1+np.sign(scores))/2).astype(int)
        #   Y_hat = self.model.predict(X)

        
        n = len(Y_hat)
        #Y_hat = np.reshape(Y_hat,n)
        

        if(self.margin_function=='dot_product'):
            margin_score = np.abs(scores) #np.array([ float(abs(np.dot(self.w,x) + self.b)) for x in X]) #.squeeze()

            if self.num_classes == 2:
                margin_score = margin_score.flatten()
            else:
                O = np.sort(out['probs'],axis=1)
                margin_score = np.abs(O[:,0]-O[:,1]) #np.abs(margin_score[Y_hat])
                #margin_score = out['probs']
        else:
            margin_score = 1/(1+np.exp(-scores)) 

        
        out['labels'] = Y_hat 
        out['confidence']=np.max(out['probs'],axis=1) 
        
        out['abs_logit'] = np.abs(margin_score)
        
        
        return out 

    def get_weights(self):
        coef_ = self.model.coef_.squeeze()

        d = len(coef_)
        
        if(self.model_conf['fit_intercept']):    
            w = np.zeros( d+ 1)
            w[:d] = self.w 
            w[d] = self.b 
            #w[:d] = self.model.coef_
            #w[d] =  self.model.intercept_
            return  w
        else:
            w = np.zeros(d)
            w[:] = self.w #self.model.coef_
            return w