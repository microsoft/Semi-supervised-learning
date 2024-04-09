import matplotlib.pyplot as plt 
import pandas as pd

c_tbal = 'red' 
c_al = 'royalblue'
c_alsc= 'deepskyblue'
c_pl = 'gray'#'coral'
c_plsc= 'slategray'#'darksalmon'

ls_tbal = '-'
ls_alsc = '-'
ls_plsc = '-'
ls_al = 'dashdot'
ls_pl = 'dashdot'  

me_tbal_err = 1 
me_alsc_err = 1 
me_plsc_err = 1 
me_al_err = 1 
me_pl_err = 1

me_tbal_cov = 1 
me_alsc_cov = 1 
me_plsc_cov = 1 
me_al_cov = 1 
me_pl_cov = 1
eps=0.01 

from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 

xrotate = 45

def h_plot(ax,x,mu,std,label,marker,color,linestyle='solid',markevery=2,alpha=1.0):
    M =1
    
    l1,=ax.plot(x,mu*M,marker=marker,label=label,color=color,linestyle=linestyle,
    markevery=markevery,linewidth=2.5,markersize=7,alpha=alpha)
    ax.fill_between(x,(mu-std)*M,(mu+std)*M,alpha=0.2,color=color)
    return l1

def gen_plot1(ax1,out,lst_x,xlabel,x_ticks,xtick_labels,eps,y_low,y_up):
    
    l1=h_plot(ax1,lst_x,out['ALBL_sel_err_mean'],out['ALBL_sel_err_std'],'TBAL',
            '*', color=c_tbal,linestyle=ls_tbal,markevery=me_alsc_err)
     
    l2=h_plot(ax1,lst_x,out['AL_sel_err_mean'],out['AL_sel_err_std'],'AL+SC',
            'd',color=c_alsc,linestyle=ls_alsc,markevery=me_alsc_err) 
    
    l3=h_plot(ax1,lst_x,out['AL_all_err_mean'],out['AL_all_err_std'],'AL',
            marker='x',color=c_al,linestyle=ls_al,markevery=me_al_err)
    
    l4=h_plot(ax1,lst_x,out['PL_sel_err_mean'],out['PL_sel_err_std'],'PL+SC',
            'v', color=c_plsc,linestyle=ls_plsc,markevery = me_plsc_err)
     
    l5=h_plot(ax1,lst_x,out['PL_all_err_mean'],out['PL_all_err_std'],'PL',
            'o',color=c_pl,linestyle=ls_pl,markevery=me_pl_err)

    l7 = ax1.axhline(eps*100,color='green',
                            linestyle='dashdot',linewidth=3.0,label='Error Threshold $\epsilon_a$')
    ax1.set_xlabel(xlabel)
    
    #
    #formatter.set_powerlimits((-1,3)) 
    #ax1.xaxis.set_major_formatter(formatter) 
    
    ax1.set_ylim(y_low,y_up)

    #ax1.set_yticks(y_ticks,minor=True)
    #ax1.set_yticks(y_ticks_major)
    ax1.set_xticks(x_ticks)
    #ax1.set_xticks(x_ticks,minor=True)
    #ax1.set_xticks(x_ticks_err_major)
    #ax1.set_xticklabels(x_ticks, rotation = xrotate )

    #ax1.set_xticklabels(ax1.get_xticks(), rotation = xrotate)
    ax1.set_xticklabels(xtick_labels)
    #ax1.yaxis.set_major_formatter(OOMFormatter(5, "%1.1f"))
    
    #ax1.ticklabel_format(axis='x',style='scientific',scilimits=(0,2000))

    ax1.set_ylabel('Auto-Labeling Error (%)')

    return [l1,l2,l3,l4,l5,l7]

def gen_plot2(ax1,out,lst_x,xlabel,x_ticks,xtick_labels,y_low,y_up):
    

    l1=h_plot(ax1,lst_x,out['ALBL_sel_cov_mean'],out['ALBL_sel_cov_std'],'TBAL',
            '*',color=c_tbal,linestyle=ls_tbal,markevery=me_tbal_cov) 
    
    l2=h_plot(ax1,lst_x,out['AL_sel_cov_mean'],out['AL_sel_cov_std'],'AL+SC',
            'd',color=c_alsc,linestyle=ls_alsc,markevery=me_alsc_cov) 
    
    l3=h_plot(ax1,lst_x,out['AL_all_cov_mean'],out['AL_all_cov_std'],'AL',
            marker='x',color=c_al,linestyle=ls_al,markevery=me_al_cov)
    
    
    l4=h_plot(ax1,lst_x,out['PL_sel_cov_mean'],out['PL_sel_cov_std'],'PL+SC',
            'v', color=c_plsc,linestyle=ls_plsc,markevery=me_plsc_cov)
     
    l5=h_plot(ax1,lst_x,out['PL_all_cov_mean'],out['PL_all_cov_std'],'PL',
            'o',color=c_pl,linestyle=ls_pl,markevery=me_pl_cov)


    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Coverage (%)')

    ax1.set_xticks(x_ticks)

    ax1.set_xticklabels(xtick_labels)
  
    #ax1.set_xticklabels(ax1.get_xticks(), rotation = xrotate)

    #ax1.set_xticks(x_ticks,minor=True)
    #ax1.set_xticks(x_ticks_err_major)

    #ax1.set_xticklabels(x_ticks, rotation = xrotate )

    ax1.set_ylim(y_low,y_up)
    
    #ax1.set_yticks(y_ticks_cov,minor=True)
    #ax1.set_yticks(y_ticks_cov_major)

    #ax1.grid(color='b', alpha=0.5, linestyle='dashed', linewidth=0.5)

    return [l1,l2,l3]

def new_plot(
        save_path: str,
        df: pd.DataFrame,
        lst_x: list,
        xlabel: str,
        x_ticks: list,
        xtick_lbls: list,
        eps: float=0.01,
        cov: int = 1,
        title: str = "",
        figsize: tuple=(10,4),
        legend: bool =False,
        y_low_1: int =0,
        y_up_1: int = 30, 
        y_low_2: int =0,
        y_up_2: int =101):


    fig,ax = plt.subplots(1,2,figsize=figsize)
    methods_symbol = [
        ('auto_label_opt_v0', '*', 'red', '-', 1), 
        ('dirichlet', 'v', 'royalblue', '-', 1), 
        ('histogram_binning_top_label', 'd', 'deepskyblue', '-', 1 ), 
        ('scaling_binning', 'o', 'purple', 'dashdot', 1), 
        ('temp_scaling', 'x', 'slategray', 'dashdot', 1), 
        ('None', 's', 'gray', 'dashdot', 1)
        ]

    ##### Plot AL Error #####
    al_err_plots = []
    for method, symbol, c, ls, me in methods_symbol:
        n_t = df[df['calib_conf'] == f'{method}']['N_t'].to_numpy()
        al_cov_mean = df[df['calib_conf'] == f'{method}']['Auto-Labeling-Err-Mean'].to_numpy()
        al_cov_std = df[df['calib_conf'] == f'{method}']['Auto-Labeling-Err-Std'].to_numpy()

        l1=h_plot(ax[0],n_t,al_cov_mean,al_cov_std,f'{method}',
                f'{symbol}',color=c,linestyle=ls,markevery=me) 
        al_err_plots.append(l1)


    l7 = ax[0].axhline(eps*100,color='green',
                            linestyle='dashdot',linewidth=3.0,label='Error Threshold $\epsilon_a$')
    al_err_plots.append(l7)

    legendEntries = [method for method, _, _, _,_ in methods_symbol]
    if(legend):
        lgd=fig.legend(al_err_plots,legendEntries,ncol=6,loc="lower center",bbox_to_anchor=(0.53, -0.06),
                borderaxespad=0, frameon=True, markerscale=2)

    ax[0].set_xlabel(xlabel)
    ax[0].set_ylim(y_low_1,y_up_1)
    ax[0].set_xticks(x_ticks)
    ax[0].set_xticklabels(xtick_lbls)
    ax[0].set_ylabel('Auto-Labeling Error (%)')

    ##### Plot AL Coverage #####
    al_coverage_plots = []
    for method, symbol, c, ls, me in methods_symbol:
        n_t = df[df['calib_conf'] == f'{method}']['N_t'].to_numpy()
        al_cov_mean = df[df['calib_conf'] == f'{method}']['Coverage-Mean'].to_numpy()
        al_cov_std = df[df['calib_conf'] == f'{method}']['Coverage-Std'].to_numpy()

        l1=h_plot(ax[1],n_t,al_cov_mean,al_cov_std,f'{method}',
                f'{symbol}',color=c,linestyle=ls,markevery=me) 
        al_coverage_plots.append(l1)


    al_coverage_plots.append(l7)

    legendEntries = [method for method, _, _, _,_ in methods_symbol]
    if(legend):
        lgd=fig.legend(al_coverage_plots,legendEntries,ncol=6,loc="lower center",bbox_to_anchor=(0.53, -0.06),
                borderaxespad=0, frameon=True, markerscale=2)

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel('Coverage (%)')
    ax[1].set_xticks(x_ticks)
    ax[1].set_xticklabels(xtick_lbls)
    ax[1].set_ylim(y_low_2,y_up_2)

    # Glob settings
    ax[0].grid(color='gray', alpha=0.4, linestyle='dashed', linewidth=0.4,which='both')
    ax[1].grid(color='gray', alpha=0.4, linestyle='dashed', linewidth=0.4,which='both')


    visualize = True 

    plt.tight_layout()

    if(title):
        plt.suptitle(title)
    if(save_path is not None and legend):
        plt.savefig(save_path,dpi=300, transparent=False,bbox_extra_artists=(lgd,), bbox_inches='tight')
    if(save_path is not None and legend == False):
        plt.savefig(save_path,dpi=300, transparent=False, bbox_inches='tight')
    if(not visualize):
        plt.close(fig)



def plot(save_path,
         out,
         lst_x,
         xlabel,
         x_ticks,
         xtick_lbls,
         eps=0.01,
         cov=1,
         title=None,
         figsize=(10,4),
         legend=False,
         y_low_1=0,
         y_up_1= 30, 
         y_low_2=0,
         y_up_2=101):
    fig,ax = plt.subplots(1,2,figsize=figsize)

    L = gen_plot1(ax[0],out,lst_x,xlabel,x_ticks,xtick_lbls, eps,y_low_1,y_up_1)
    
    M = gen_plot2(ax[1],out,lst_x,xlabel,x_ticks,xtick_lbls,y_low_2,y_up_2)

    legendEntries = ['TBAL ','AL+SC','AL','PL+SC','PL','$\epsilon_a$']

    ax[0].grid(color='gray', alpha=0.4, linestyle='dashed', linewidth=0.4,which='both')
    ax[1].grid(color='gray', alpha=0.4, linestyle='dashed', linewidth=0.4,which='both')

    if(legend):
        lgd=fig.legend(L,legendEntries,ncol=6,loc="lower center",bbox_to_anchor=(0.53, -0.06),
                borderaxespad=0, frameon=True, markerscale=2)

    visualize = True 

    plt.tight_layout()

    #plt.subplots_adjust(top=0.80)
    if(title):
        plt.suptitle(title)
    #plt.subplots_adjust(top=0.)     # Add space at top
    #plt.subplots_adjust(right=1.1)
    if(save_path is not None and legend):
        plt.savefig(save_path,dpi=300, transparent=False,bbox_extra_artists=(lgd,), bbox_inches='tight')
    if(save_path is not None and legend == False):
        plt.savefig(save_path,dpi=300, transparent=False, bbox_inches='tight')
    if(not visualize):
        plt.close(fig)


