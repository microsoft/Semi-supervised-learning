# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import re
import operator
import numpy as np
save_path = r'../saved_models/'
static_dict = {}

def get_static(file_name):
    re_bestAcc = r'BEST_EVAL_ACC: (([0-9]|\.)*)'  # .group(1)
    re_bestIt = r'at ([0-9]*)'  # .group(1)
    re_top1Acc = r"eval\/top-1-acc': (([0-9]|\.)*)"
    re_top5Acc = r"eval\/top-5-acc': (([0-9]|\.)*)"
    
    stat = {"bestAcc": 0,
            "bestIt": 0,
            "Top1Acc": [],
            "Top5Acc": [],
            }
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        continue_flag = False
        for line in lines:
            if '1048000 iteration' in line:
                continue_flag = True
    if continue_flag == False:
        return {'Top1_1': [],
                'Top1_20': [],
                'Top1_50': [],
                'Top5_1': [],
                'Top5_20': [],
                'Top5_50': [],
                'BestAcc': 0,
                'BestIt': 0,
                'Finish': False}
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.endswith('iters\n'):
                stat['bestAcc'] = re.search(re_bestAcc,line).group(1)
                stat['bestIt'] = re.search(re_bestIt,line).group(1)
                stat['Top1Acc'].append(re.search(re_top1Acc,line).group(1))
                stat['Top5Acc'].append(re.search(re_top5Acc,line).group(1))
    for i in range(len(stat['Top1Acc'])):
        stat['Top1Acc'][i] = float(stat['Top1Acc'][i])
    for i in range(len(stat['Top5Acc'])):
        stat['Top5Acc'][i] = float(stat['Top5Acc'][i])
    stat['bestAcc'] = float(stat['bestAcc'])
    avg_1_1acc = stat['Top1Acc'][-1]
    avg_20_1acc = sum(stat['Top1Acc'][-20:])/20
    avg_50_1acc = sum(stat['Top1Acc'][-50:])/50
    avg_1_5acc = stat['Top5Acc'][-1]
    avg_20_5acc = sum(stat['Top5Acc'][-20:])/ 20
    avg_50_5acc = sum(stat['Top5Acc'][-50:])/ 50
    return {'Top1_1': avg_1_1acc,
            'Top1_20': avg_20_1acc,
            'Top1_50': avg_50_1acc,
            'Top5_1': avg_1_5acc,
            'Top5_20': avg_20_1acc,
            'Top5_50': avg_50_1acc,
            'BestAcc': stat['bestAcc'],
            'BestIt': stat['bestIt'],
            'Finish': True}

# str = r"[2021-04-13 15:57:33,078 INFO] 228000 iteration, USE_EMA: True, {'train/sup_loss': tensor(0.0311, device='cuda:0'), 'train/unsup_loss': tensor(0.2391, device='cuda:0'), 'train/total_loss': tensor(0.3913, device='cuda:0'), 'train/mask_ratio': tensor(0.5246, device='cuda:0'), 'lr': 0.028670201217471786, 'train/prefetch_time': 0.0050832958221435545,'train/run_time': 0.315829833984375, 'eval/loss': tensor(1.0763, device='cuda:0'), 'eval/top-1-acc': 0.6306},BEST_EVAL_ACC: 0.9348, at 173000 iters"

statics = {}
for name in os.listdir(save_path):
    cur_path = save_path + name
    if os.path.isdir(cur_path):
        cur_name = name
        for n in os.listdir(cur_path):
            if n == 'log.txt':
                #try:
                #    statics[cur_name] = get_static(cur_path + '/' + n)
                #except:
                #    print(cur_path,'failed')
                statics[cur_name] = get_static(cur_path + '/' + n)
statics = sorted(statics.items())
final_res = {}
for s in statics:
    exp_name = '_'.join(s[0].split('_')[:-1])
    if s[1]['Finish'] == False:
        print(s[0], 'is not finished')
        continue
    if exp_name not in final_res:
        tmp = {'Top1_1': [s[1]['Top1_1']*100],
            'Top1_20': [s[1]['Top1_20']*100],
            'Top1_50': [s[1]['Top1_50']*100],
            'Top5_1': [s[1]['Top5_1']*100],
            'Top5_20': [s[1]['Top5_20']*100],
            'Top5_50': [s[1]['Top5_50']*100],
            'BestAcc': [s[1]['BestAcc']*100]}
        final_res[exp_name] = tmp
    else:
        final_res[exp_name]['Top1_1'].append(s[1]['Top1_1']*100)
        final_res[exp_name]['Top1_20'].append(s[1]['Top1_20']*100)
        final_res[exp_name]['Top1_50'].append(s[1]['Top1_50']*100)
        final_res[exp_name]['Top5_1'].append(s[1]['Top5_1']*100)
        final_res[exp_name]['Top5_20'].append(s[1]['Top5_20']*100)
        final_res[exp_name]['Top5_50'].append(s[1]['Top5_50']*100)
        final_res[exp_name]['BestAcc'].append(s[1]['BestAcc']*100)
#for k,v in final_res.items():
    #print(k,'Last 01 epoch, Top1 acc, mean: ',np.mean(v['Top1_1']), ', std: ', np.std(v['Top1_1']))
    #print(k,'Last 20 epoch, Top1 acc, mean: ',np.mean(v['Top1_20']), ', std: ', np.std(v['Top1_20']))
    #print(k,'Last 50 epoch, Top1 acc, mean: ',np.mean(v['Top1_50']), ', std: ', np.std(v['Top1_50']))
    #print(k,'Last 01 epoch, Top5 acc, mean: ',np.mean(v['Top5_1']), ', std: ', np.std(v['Top5_1']))
    #print(k,'Last 20 epoch, Top5 acc, mean: ',np.mean(v['Top5_20']), ', std: ', np.std(v['Top5_20']))
    #print(k,'Last 50 epoch, Top5 acc, mean: ',np.mean(v['Top5_50']), ', std: ', np.std(v['Top5_50']))
    #print(k,'Best epoch, Top1 acc, mean: ',np.mean(v['BestAcc']), ', std: ', np.std(v['BestAcc']))

import xlwt

data_setting=['cifar10_40','cifar10_250','cifar10_4000','cifar100_400','cifar100_2500','cifar100_10000','stl10_40','stl10_250','stl10_1000','svhn_40','svhn_250','svhn_1000']
show_acc = ['BestAcc','Top1_1','Top1_20','Top1_50','Top5_1','Top5_20','Top5_50']
algs = ['pimodel','pseudolabel','pseudolabel_flex','meanteacher','vat','mixmatch','remixmatch','uda','uda_flex','fixmatch','flexmatch','fullysupervised']
workbook = xlwt.Workbook()
for i in range(len(show_acc)):
    worksheet = workbook.add_sheet(show_acc[i],cell_overwrite_ok=True)
    for j in range(len(algs)):
        worksheet.write(j+1,0,algs[j])
    for j in range(len(data_setting)):
        for alg in algs:
            worksheet.write(algs.index(alg)+1,j+1,'None')
            
    for j in range(len(data_setting)):
        d = data_setting[j]
        worksheet.write(0,j+1,d)
        for alg in algs:
            for k,v in final_res.items():
                if alg+'_'+d == k and show_acc[i] in v:
                #if k.endswith('_'+d)  and k.startswith(alg+'_') and show_acc[i] in v:
                    worksheet.write(algs.index(alg)+1,j+1,str(round(np.mean(v[show_acc[i]]),2))+u"\u00B1"+str(round(np.std(v[show_acc[i]]),2)))
                
workbook.save('../saved_models/final_res.xls')
