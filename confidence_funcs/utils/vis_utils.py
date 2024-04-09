import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import animation
from IPython.display import HTML
import matplotlib.image as mpimg 
from math import atan, cos, sin,acos,pi,sqrt

import numpy as np
import matplotlib.lines as mlines
import os 

def visualize_db(classifier, X=None,Y=None,x_min=None,x_max=None,y_min=None,y_max=None):
    if(X is not None and Y is not None):
        x_min,x_max = X[:,0].min(), X[:,0].max()
        y_min,y_max = X[:,1].min(), X[:,1].max()
    x_min = x_min *1.2
    y_min = y_min *1.2
    x_max = x_max*1.2
    y_max = y_max*1.2
    
    xx, yy = np.mgrid[x_min:x_max:.01, y_min:y_max:.01]

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = classifier.predict_proba(grid)
    probs = probs[:,1].reshape(xx.shape)

    f, ax = plt.subplots(figsize=(8, 6))

    contour = ax.contour(xx, yy, probs, 1, cmap="Greys", vmin=0.0, vmax=0.9)
    sns.scatterplot(x =X[:,0],y=X[:,1],hue=(Y+1)/2,ax=ax)
    return ax


class UnitCircleCanvas:
    
    def __init__(self,r,ax):
        ax = plt.gca()
        ax.cla() 
        circle1 = plt.Circle((0, 0), r, color='b', fill=False)
        circle2 = plt.Circle((0, 0), 0.025, color='black', fill=False)
        ax.set_xlim((-r-0.2, r+0.2))
        ax.set_ylim((-r-0.2, r+0.2))
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        
        self.r = r
        self.ax = ax
    
    def unit_vector_at_angle(self,w,theta):
        r1 = 1
        theta_0 = atan(w[1]/w[0])
        theta_1 = theta_0 + theta
        w2 = np.array([ r1* cos(theta_1) , r1*sin(theta_1)])
        return w2
    
    def draw_line(self,w,b,color='b',linestyle='-',lw=2.,label=None):
        x1,x2 = self.line_end_points_on_circle_2d(w[0],w[1],b)
        #print(x1,x2)
        x_arr = np.arange(x1-1e-5,x2+1e-5,1e-5)
        m     = w[0]/w[1]
        y1    = -m*x_arr - b/w[1] 
        if(label is None):
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle)
        else:
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle,label=label)
        self.ax.add_line(line1)
        
    
    def draw_line_at_angle(self,w,b,theta,color='b',linestyle='-',lw=2.,label=None):
        w2 = self.unit_vector_at_angle(w,theta)
        #print(w2)
        self.draw_line(w2,b,color=color,linestyle=linestyle,lw=lw,label=label)
    
    def line_end_points_on_circle_2d(self,w0,w1,b):
        # w_0x + w_1y +b =0
        # x^2 + y^2 = r^2
        m1 = w0/w1
        m2 = b/w1
        a = 1 + m1**2
        b = 2*m1*m2
        c = m2**2 - self.r**2
        d = b**2 - 4*a*c
        if(d<1e-6):
            d = 0
        d = sqrt(d)
        x1 = (-b-d)/(2*a)
        x2 = (-b+d)/(2*a)
        return x1,x2
    

class RectangleCanvas:
    
    def __init__(self,x_min,x_max,y_min,y_max,ax):
        ax = plt.gca()
        ax.cla() 
        
        ax.set_xlim((x_min, x_max))
        ax.set_ylim((y_min, y_max))
        #ax.add_patch(circle1)
        #ax.add_patch(circle2)
        
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.x_min = x_min

        self.ax = ax
    
    def unit_vector_at_angle(self,w,theta):
        r1 = 1
        theta_0 = atan(w[1]/w[0])
        theta_1 = theta_0 + theta
        w2 = np.array([ r1* cos(theta_1) , r1*sin(theta_1)])
        return w2
    
    def draw_line(self,w,b,color='b',linestyle='-',lw=2.,label=None):
        #x1,x2 = self.line_end_points_on_circle_2d(w[0],w[1],b)
        #print(x1,x2)
        x_arr = np.arange(self.x_min, self.x_max, 1e-5)
        m     = w[0]/w[1]
        y1    = -m*x_arr - b/w[1] 
        if(label is None):
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle)
        else:
            line1 = mlines.Line2D(x_arr,y1, lw=lw, alpha=0.3,color=color,linestyle=linestyle,label=label)
        self.ax.add_line(line1)
        
    
    def draw_line_at_angle(self,w,b,theta,color='b',linestyle='-',lw=2.,label=None):
        w2 = self.unit_vector_at_angle(w,theta)
        #print(w2)
        self.draw_line(w2,b,color=color,linestyle=linestyle,lw=lw,label=label)
    

def split_pos_neg_pts(lst_idx,Y):
    qp = []
    qn = []
    for i in lst_idx:
        if(Y[i]==1):
            qp.append(i)
        else:
            qn.append(i)
    return qp,qn
    
def vis_epoch_learning(per_epoch_out,epoch,true_db,ds,save_path,visualize=True,canvas='circle'):
    epoch_out = per_epoch_out[epoch]
    X,Y = ds.X,ds.Y 

    plt.figure(figsize=(8.5,8.5))
    fig,ax = plt.subplots(figsize=(8,8))
    if(canvas=='circle'):
        u_cnvs = UnitCircleCanvas(1,ax)
    else:
        x_min,x_max = min(X[:,0])-0.5, max(X[:,0])+0.5
        y_min, y_max = min(X[:,1]) -0.5, max(X[:,1]) +0.5
        u_cnvs = RectangleCanvas(x_min,x_max,y_min,y_max,ax)
    
    #w_star= w_star/np.linalg.norm(w_star)
    #u_cnvs.draw_line(w_star,0,color='red',lw=3.,label='$w^*$')

    db = true_db 
    ax.scatter(db[:,0],db[:,1],s=1,color='red')

    w = epoch_out['clf_weights']
    b = 0
    if(len(w)==3):
        w_hat = w[:2]
        #print(np.linalg.norm(w_hat), w[2])
        b = w[2]
    #w_hat = w_hat/ (np.linalg.norm(w_hat))
    
    u_cnvs.draw_line(w_hat,b,color='green',lw=3.,label='$\hat{w}_k$')
    
    # do it only for margin sampling...

    #u_cnvs.draw_line(w_hat,b_k,color='black',lw=1.,linestyle='dashed')
    #u_cnvs.draw_line(w_hat,-b_k,color='black',lw=1.,linestyle='dashed')
    
    all_train_pts_till_epoch =[]
    #all_train_pts_till_epoch.extend(per_epoch_out[0]['selected_points_idx'])
    for e in range(epoch):
        all_train_pts_till_epoch.extend(per_epoch_out[e]['query_points'])
    
    q = epoch_out['query_points']
  
    qp,qn = split_pos_neg_pts(q,Y)
    prev_qp, prev_qn = split_pos_neg_pts(all_train_pts_till_epoch,Y)
    
    ax.scatter(X[:,0],X[:,1],color='gray', s=1.0)
    ax.scatter(X[prev_qp,0],X[prev_qp,1],s=4.0,color='blue',label='prev. +ve queries')
    ax.scatter(X[prev_qn,0],X[prev_qn,1],s=4.0,color='red',label='prev. -ve queries')
    
    ax.scatter(X[qp,0],X[qp,1],s=18.0,marker='x',color='blue',label='current +ve queries')
    ax.scatter(X[qn,0],X[qn,1],s=18.0,marker='x',color='red',label='current -ve queries')
    
  
    ax.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(save_path)
    if(not visualize):
        plt.close(fig)
        

def vis_all_margins(per_epoch_out,epoch,w_star,X,Y,save_path,visualize=True,canvas='circle'):
    epoch_out = per_epoch_out[epoch]
    
    plt.figure(figsize=(8.5,8.5))
    fig,ax = plt.subplots(figsize=(8,8))
    if(canvas=='circle'):
        u_cnvs = UnitCircleCanvas(1,ax)
    else:
        x_min,x_max = min(X[:,0])-0.5, max(X[:,0])+0.5
        y_min, y_max = min(X[:,1]) -0.5, max(X[:,1]) +0.5
        u_cnvs = RectangleCanvas(x_min,x_max,y_min,y_max,ax)
    
    w_star= w_star/np.linalg.norm(w_star)
    u_cnvs.draw_line(w_star,0,color='red',lw=1.,label='$w^*$')
    
    #w_hat = epoch_out['clf_weights']
    #w_hat = w_hat/ (np.linalg.norm(w_hat))
    
    #u_cnvs.draw_line(w_hat,0,color='green',lw=3.,label='$\hat{w}_k$')
    colors = ['black','green','blue']
    for i in range(epoch+1):
        # do it only for margin sampling...
        t_hat_i = per_epoch_out[i]['t_i']
        w_hat =  per_epoch_out[i]['clf_weights']
        w_hat = w_hat/ (np.linalg.norm(w_hat))

        print(t_hat_i)
        u_cnvs.draw_line(w_hat,t_hat_i,color=colors[i],lw=1.,linestyle='dashed')
        u_cnvs.draw_line(w_hat,-t_hat_i,color=colors[i],lw=1.,linestyle='dashed')
    


def vis_epoch_labeling(per_epoch_out,epoch,true_db,ds,save_path,visualize=True,canvas='circle'):
    epoch_out = per_epoch_out[epoch]
    X,Y = ds.X, ds.Y

    plt.figure(figsize=(8.5,8.5))
    fig,ax = plt.subplots(figsize=(8,8))
    if(canvas=='circle'):
        u_cnvs = UnitCircleCanvas(1,ax)
    else:
        x_min,x_max = min(X[:,0])-0.5, max(X[:,0])+0.5
        y_min, y_max = min(X[:,1]) -0.5, max(X[:,1]) +0.5
        u_cnvs = RectangleCanvas(x_min,x_max,y_min,y_max,ax)
    
    #w_star= w_star/np.linalg.norm(w_star)
    #u_cnvs.draw_line(w_star,0,color='red',lw=3.,label='$w^*$')
    db = true_db 
    ax.scatter(db[:,0],db[:,1],s=1,color='red')

    
    w = epoch_out['clf_weights']
    b = 0
    if(len(w)==3):
        w_hat = w[:2]
        #print(np.linalg.norm(w_hat), w[2])
        b = w[2]
    #w_hat = w_hat/ (np.linalg.norm(w_hat))
    
    u_cnvs.draw_line(w_hat,b,color='green',lw=3.,label='$\hat{w}_k$')
    
    # do it only for margin sampling...
    t_hat_i = epoch_out['lst_t_i']
    
    u_cnvs.draw_line(w_hat,max(-0.95,b+t_hat_i[0]),color='black',lw=2.,linestyle='dashed')
    u_cnvs.draw_line(w_hat,min(0.95,b-t_hat_i[1]),color='black',lw=2.,linestyle='dashed')
    

    all_train_pts_till_epoch =[]
    all_train_pts_till_epoch.extend(per_epoch_out[0]['seed_train_pts'])
    for e in range(epoch):
        all_train_pts_till_epoch.extend(per_epoch_out[e]['query_points'])
    
    prev_qp, prev_qn = split_pos_neg_pts(all_train_pts_till_epoch,Y)
    ax.scatter(X[prev_qp,0],X[prev_qp,1],s=6.0,color='blue',label='prev. +ve queries')
    ax.scatter(X[prev_qn,0],X[prev_qn,1],s=6.0,color='red',label='prev. -ve queries')
    

    if('query_points' in epoch_out):
        q = epoch_out['query_points']
        qp,qn = split_pos_neg_pts(q,Y)
        ax.scatter(X[qp,0],X[qp,1],s=30.0,marker='x',color='blue',label='current +ve queries')
        ax.scatter(X[qn,0],X[qn,1],s=30.0,marker='x',color='red',label='current -ve queries')

    
    #
    #ax.scatter(X[:,0],X[:,1],color='gray', s=1.0)
    
    # show unlabeled pts in gray
    if(epoch<len(per_epoch_out)-1):
        unlabeled_pts = np.array(per_epoch_out[epoch+1]['unlabeled_pts_idcs'])
        X_u = X[unlabeled_pts]
        ax.scatter(X_u[:,0],X_u[:,1],marker='x',color='black', s=2.0 ,label='unlabeled ')
    

    #epoch_out['query_points']

    show_auto_lbld = True
    if(show_auto_lbld):
        #ax.scatter(X[:,0],X[:,1],marker='o',s=0.2,color='black')
        auto_lbleled_pts_till_now = [] 
        #auto_labeled_pts = epoch_out['auto_lbld_idx_lbl']
        
        for e in range(epoch+1):
            auto_lbleled_pts_till_now.extend(per_epoch_out[e]['auto_lbld_idx_lbl'])

        auto_lbl_idx = [x[0] for x in auto_lbleled_pts_till_now]
        auto_lbl_Y   = [x[1] for x in auto_lbleled_pts_till_now]
        num_auto = len(auto_lbl_Y)
        qp = [ auto_lbl_idx[i] for i in range(num_auto) if int(auto_lbl_Y[i]) ==1 ]
        qn = [ auto_lbl_idx[i] for i in range(num_auto) if int(auto_lbl_Y[i]) ==0 ]
        print(len(qp),len(qn))
        ax.scatter(X[qp,0],X[qp,1],s=2.0,marker='.',color='blue',label='auto-labeled +ve')
        ax.scatter(X[qn,0],X[qn,1],s=2.0,marker='.',color='red', label='auto-labeled -ve')
    
    show_val = False 
    if(show_val):
        val_idcs = np.array(per_epoch_out[epoch]['end_val_idcs'])
        X_v = X[val_idcs]
        ax.scatter(X_v[:,0],X_v[:,1],color='purple', s=2.0 ,label='validation ')
    
    #qp,qn = split_pos_neg_pts(list(range(len(X))),Y)
    #ax.scatter(X[qp,0],auto_lbl_Y[qp,1],s=0.1,marker='o',color='blue')
    #ax.scatter(X[qn,0],auto_lbl_Y[qn,1],s=0.1,marker='o',color='red')
    #ax.set_title('')
  
    ax.legend(loc='center right')
    plt.tight_layout()
    plt.savefig(save_path)
    if(not visualize):
        plt.close(fig)
    
# mode : labeling or learning
def animate(mode,lst_epoch_out,true_db,ds,plots_directory,canvas,interval=1000):
    plt.rcParams['figure.figsize'] = (6,6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 100
    plt.rcParams["animation.html"] = "jshtml"  # for matplotlib 2.1 and above, uses JavaScript
    #plt.rcParams["animation.html"] = "html5" # for matplotlib 2.0 and below, converts to x264 using ffmpeg video codec
    fig, ax = plt.subplots(figsize=(6,6))

    #l, = ax.plot([],[])
    t  = min(len(lst_epoch_out),50)
    title_head = ''
    def animate(i):
        title_head = ''
        if('clf_weights' in lst_epoch_out[i]):
            if(mode == 'labeling'):
                vis_epoch_labeling(lst_epoch_out,i,true_db,ds,'{}/{}.png'.format(plots_directory,i),visualize=False,canvas=canvas)
                title_head = 'Active Labeling'
            elif(mode == 'learning'):
                vis_epoch_learning(lst_epoch_out,i,true_db,ds,'{}/{}.png'.format(plots_directory,i),visualize=False,canvas=canvas)
                title_head = 'Active Learning'
        
            fpath = '{}/{}.png'.format(plots_directory,i)
            img = mpimg.imread(fpath)
            ax.imshow(img)
            ax.set_title('{} Epoch (k) : {}'.format(title_head, i))
            ax.axis('off')

    ani = animation.FuncAnimation(fig, animate, frames=t ,interval=interval)
    plt.tight_layout()
    plt.close()
    return ani 

def vis_dataset(dataset):
    Y = dataset.Y 
    X = dataset.X 
    idx_p = np.where(Y==1)[0]
    idx_n = np.where(Y==0)[0]
    plt.figure(figsize=(6,6))
    plt.scatter(X[idx_p,0],X[idx_p,1],s=1)
    plt.scatter(X[idx_n,0],X[idx_n,1],s=1)