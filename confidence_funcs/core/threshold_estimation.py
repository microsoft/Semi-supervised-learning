
from sklearn.metrics import accuracy_score
import numpy as np 

def determine_threshold(lst_classes,inf_out,auto_lbl_conf,val_ds,val_idcs,logger,err_threshold=0.01):
    
    # run inference on the validation points.
    # inference will give, label + conf for each poitn
    # for each conf find the accuracy on points with conf >= this conf
    # settle for the lowest conf that satisifies the accuracy requirement
    # if no conf found, return t_max
    max_t = float('inf')

    Y_val   = val_ds.Y 
    y_hat   = inf_out['labels']
    
    n_v = len(y_hat)

    score_type = auto_lbl_conf['score_type']

    scores = inf_out[score_type] 
    
    val_err = 1-accuracy_score(Y_val,y_hat)
    fast = False
    

    if(fast and len(scores)>500):
        #print('here')
        min_score =min(scores)
        max_score = max(scores)
        delta = (max_score-min_score)/20000

        logger.debug("MAX score = {}, MIN score = {}, delta = {}".format(max_score,min_score,delta))
        
        if abs(max_score - min_score) <= 1e-8:
            lst_th = [min_score]
        else:
            lst_th = np.arange(min_score,max_score+delta,delta)
    else:
        lst_th = scores 
    
    lst_th = np.array(lst_th)

    S = np.zeros((n_v,8))

    S[:,0] = np.arange(0,n_v,1)
    S[:,1] = Y_val 
    S[:,2] = y_hat
    S[:,3] = S[:,1]==S[:,2]
    S[:,4] = inf_out['confidence']
    #S[:,5] = inf_out['logits']
    S[:,6] = inf_out['abs_logit']
    
    
    if('energy_score' in inf_out):
        S[:,7] = inf_out['energy_score']

    scores_id_map = {'confidence':4,'logits':5,'abs_logit': 6,'energy_score':7}
    
    score_key = scores_id_map[score_type]
    
    # sort in descending order of score
    S = S[(-S[:,score_key]).argsort()] 

    def get_err_at_th(S_y,c):            
        S2 = S_y[S_y[:,score_key]>=c]
        if(len(S2)>0):
            return 1-(S2[:,3].sum()/len(S2))
        else:
            return 1.0

    def get_std_at_th(S_y,c):            
        S2 = S_y[S_y[:,score_key]>=c]
        if(len(S2)>0):
            #print(S2.shape)
            #print(S2[:,3].shape)
            z = np.std(1-(S2[:,3]))
            #print(z)
            return np.std(1-(S2[:,3]))
        else:
            return 0
    
    C_1 = auto_lbl_conf['C_1']
    ucb = auto_lbl_conf['ucb']
    
    logger.debug(f'C_1 = {C_1} UCB = {ucb}')  

    def get_threshold(S_y):

        #std_th = np.array(std_th)
        #print(err_th)
        n_v_t = np.array([len(S_y[S_y[:,score_key]>=th]) for th in lst_th])

        n_v_0 = 10 
        
        lst_th_ = lst_th[np.where(n_v_t>n_v_0)]
        n_v_t_ = n_v_t[np.where(n_v_t>n_v_0)]

        err_th = [get_err_at_th(S_y,th) for th in lst_th_]

        #std_th = [get_std_at_th(S_y,th) for th in lst_th]
        
        err_th = np.array(err_th)
        
        #n_v_t = np.array([len(S_y[S_y[:,score_key]>=th]) for th in lst_th]) +10
        #err_th = err_th + C_1*np.sqrt(1/n_v_t)
        if(ucb=='hoeffding'):
            err_th = err_th + C_1*np.sqrt(1/n_v_t_)

        elif(ucb=='sigma'):
            err_th = err_th + C_1*np.sqrt(err_th*(1-err_th))

        #err_th = err_th + C_1*np.sqrt(err_th*(1-err_th))
        #err_th =  err_th + 2*std_th
        
        good_th = lst_th_[np.where(err_th<=err_threshold)]
        if(len(good_th)>0):
            t_y = np.min(good_th)
        else:
            t_y = max_t
        return t_y 

    
    val_idcs_to_rm = []
    lst_t_y = []
    class_wise = auto_lbl_conf['class_wise']

    if(class_wise == 'independent'):
        for y in lst_classes:
            S_y = S[S[:,2]==y]
            #print(len(S_y),S_y[0])
            t_y = get_threshold(S_y) 
            lst_t_y.append(t_y)
            logger.info('auto-labeling threshold t_i={} for class {}   '.format(t_y,y))

            if(t_y<max_t):
                idcs_vals_rm = [val_idcs[i]  for i in range(n_v) if y_hat[i] == y and scores[i]>=t_y]
                val_idcs_to_rm.extend(idcs_vals_rm)
            
            

    elif(class_wise =='joint'):
        t_ = get_threshold(S) 
        lst_t_y = [t_]*(len(lst_classes))

        logger.info('auto-labeling threshold t={} for each class.  '.format(t_))

        if(t_<max_t):
            val_idcs_to_rm = [val_idcs[i]  for i in range(n_v) if scores[i]>=t_]
    
    cov = len(val_idcs_to_rm)/len(val_idcs)
    
    logger.info(f'coverage while threshold estimation : {cov}')
    
    return lst_t_y, val_idcs_to_rm, val_err, cov 
