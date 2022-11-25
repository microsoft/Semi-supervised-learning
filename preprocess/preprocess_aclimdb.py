import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
import os
import re
import pickle
import json
import shutil
import time
import random

MODE = 'dev'
# Load translation model

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')


en2de.cuda()
de2en.cuda()

en2ru.cuda()
ru2en.cuda()



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname

def read_aclImdb(src_path='../ssl_nlp_datasets/aclImdb/',data_type='train'):
    label_dic = {'pos':0,'neg':1}
    ori_sen = []
    label = []
    if data_type != 'extra':
        for p in ['pos','neg']:
            for i in findAllFile(os.path.join(src_path,data_type,p)):
                with open(i,'r') as f:
                    tmp = f.read()
                    ori_sen.append(tmp.replace('<br />','').replace('\n',''))
                    label.append(label_dic[p])
        rand_i = list(range(len(ori_sen)))
        ori_sen, label = np.array(ori_sen),np.array(label)
        random.shuffle(rand_i)
        ori_sen, label = list(ori_sen[rand_i]), list(label[rand_i])
        print('aclIMDB originally has ',len(label), data_type, ' data!')
        return ori_sen, label
    elif data_type == 'extra':#extra is for unlabeled data. We do not use it.
        for p in ['unsup']:
            for i in findAllFile(os.path.join(src_path,'train',p)):
                with open(i,'r') as f:
                    tmp = f.read()
                    ori_sen.append(tmp.replace('<br />','').replace('\n',''))
        rand_i = list(range(len(ori_sen)))
        ori_sen = np.array(ori_sen)
        random.shuffle(rand_i)
        ori_sen = list(ori_sen[rand_i])
        return ori_sen 

def read_TextClassificationDatasets(src_path='../ssl_nlp_datasets/TextClassificationDatasets/',data_set='ag_news_csv',data_type='train'):
    ori_sen = []
    label = []
    with open(os.path.join(src_path,data_set,data_type+'.csv'),'r') as f:
        for line in f.readlines():
            if data_set == 'dbpedia_csv':
                ori_sen.append((re.split(',\"', line)[1]+re.split(',\"', line)[2]).replace('\n','')[:-1])
                label.append(re.split(',\"', line)[0])
            elif data_set == 'yelp_review_full_csv' or data_set == 'yelp_review_polarity_csv':
                ori_sen.append(re.split('\",\"', line)[1].replace('\n','')[:-1])
                label.append(re.split('\",\"', line)[0][1:])
            else:
                ori_sen.append((re.split('\",\"', line)[1]+re.split('\",\"', line)[2]).replace('\n','')[:-1])
                label.append(re.split('\",\"', line)[0][1:])
        rand_i = list(range(len(ori_sen)))
        ori_sen, label = np.array(ori_sen),np.array(label)
        random.shuffle(rand_i)
        ori_sen, label = list(ori_sen[rand_i]), list(label[rand_i])
        print(data_set, ' originally has ', len(label), data_type, ' data!')
    return ori_sen, label

def select_data(dataset,ori_sen,label,data_type='test', train_num_per_class = 5000,dev_num_per_class = 2000):
    if dataset == 'aclImdb':
        train_num_per_class, dev_num_per_class= 11500, 1000
    elif dataset == 'ag_news_csv':
        train_num_per_class, dev_num_per_class= 25000, 2500
    elif dataset == 'amazon_review_full_csv':
        train_num_per_class, dev_num_per_class = 50000, 5000
    elif dataset == 'dbpedia_csv':
        train_num_per_class, dev_num_per_class = 10000, 1000 
    elif dataset == 'yahoo_answers_csv':
        train_num_per_class, dev_num_per_class = 50000, 5000
    elif dataset == 'yelp_review_full_csv':
        train_num_per_class, dev_num_per_class = 50000, 5000
    if isinstance(label[0],str) == True:
        for i in range(len(label)):
            label[i] = int(label[i])
    if min(label) == 1:
        for i in range(len(label)):
            label[i] = label[i] - 1
    for i in range(len(label)):
        label[i] = str(label[i])
    if data_type == 'test':
        assert len(ori_sen)==len(label)
        print(dataset,' has ', len(ori_sen), ' test data!')
        return ori_sen,label
    label_dic = list(set(label))
    train_idx = {}
    train_sen = []
    train_label = []
    dev_idx = {}
    dev_sen = []
    dev_label = []
    for i in range(len(label_dic)):
        train_idx[label_dic[i]] = []
        dev_idx[label_dic[i]] = []
    for i in range(len(label)):
        for j in range(len(label_dic)):
            if label[i] == label_dic[j] and len(train_idx[label_dic[j]]) < train_num_per_class:
                train_idx[label_dic[j]].append(i)
                train_sen.append(ori_sen[i])
                train_label.append(label[i])
            elif label[i] == label_dic[j] and len(dev_idx[label_dic[j]]) < dev_num_per_class and i not in dev_idx[label_dic[j]]:
                dev_idx[label_dic[j]].append(i)
                dev_sen.append(ori_sen[i])
                dev_label.append(label[i])
    assert len(train_sen)==len(train_label)
    assert len(dev_sen)==len(dev_label)
    print(dataset,' has ', len(train_sen), ' train data and ',len(dev_sen),' dev data!')
    return train_sen,train_label,dev_sen,dev_label

def cut_sentence(s):
    # remove the first 100 words
    s = s[::-1].rsplit(' ',100)[0][::-1]
    return s


def truncate_sentence(s, max_length=1024):
    # Only keep the last max_length words
    return s[-max_length: ]


def make_json_file(ori_sen,label,dst_path='../ssl_nlp_datasets/aclImdb/',data_type='train'):
    if MODE == 'test':
        ori_sen = ori_sen[:30]
        if label is not None:
            label = label[:30]
        batchsize = 10
    else:
        batchsize = 1024 
    data = {}
    idx = 0
    if data_type == 'train':
        ori_sen_list = list(chunks(ori_sen,batchsize))
        label_list = list(chunks(label,batchsize))
        for i in tqdm(range(len(ori_sen_list))):
            cur_ori_sen = ori_sen_list[i]
            cur_label = label_list[i]
            flag = True
            while flag:
                try:
                    cur_ori_sen = list(map(truncate_sentence, cur_ori_sen))
                    cur_aug_sen_0 = de2en.translate(en2de.translate(cur_ori_sen,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
                    cur_aug_sen_1 = ru2en.translate(en2ru.translate(cur_ori_sen,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
                    flag = False
                except:
                    longest_idx = cur_ori_sen.index(max(cur_ori_sen, key = len))
                    shorter_sentence = cut_sentence(cur_ori_sen[longest_idx])
                    cur_ori_sen[longest_idx] = shorter_sentence
            for j in range(len(cur_ori_sen)):
                data[str(idx)]={}
                data[str(idx)]['ori'] = cur_ori_sen[j]
                data[str(idx)]['aug_0'] = cur_aug_sen_0[j]
                data[str(idx)]['aug_1'] = cur_aug_sen_1[j]
                data[str(idx)]['label'] = cur_label[j]
                idx = idx + 1
    elif data_type == 'test' or data_type == 'dev':
        ori_sen_list = list(chunks(ori_sen,batchsize))
        label_list = list(chunks(label,batchsize))
        for i in tqdm(range(len(ori_sen_list))):
            cur_ori_sen = ori_sen_list[i]
            cur_label = label_list[i]
            for j in range(len(cur_ori_sen)):
                data[str(idx)]={}
                data[str(idx)]['ori'] = cur_ori_sen[j]
                data[str(idx)]['label'] = cur_label[j]
                idx = idx + 1
    elif data_type == 'extra':
        ori_sen_list = list(chunks(ori_sen,batchsize))
        for i in tqdm(range(len(ori_sen_list))):
            cur_ori_sen = ori_sen_list[i]
            flag = True
            while flag:
                try:
                    cur_ori_sen = list(map(truncate_sentence, cur_ori_sen))
                    cur_aug_sen_0 = de2en.translate(en2de.translate(cur_ori_sen,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
                    cur_aug_sen_1 = ru2en.translate(en2ru.translate(cur_ori_sen,  sampling = True, temperature = 0.9),  sampling = True, temperature = 0.9)
                    flag = False
                except:
                    longest_idx = cur_ori_sen.index(max(cur_ori_sen, key = len))
                    shorter_sentence = cut_sentence(cur_ori_sen[longest_idx])
                    cur_ori_sen[longest_idx] = shorter_sentence
            for j in range(len(cur_ori_sen)):
                data[str(idx)]={}
                data[str(idx)]['ori'] = cur_ori_sen[j]
                data[str(idx)]['aug_0'] = cur_aug_sen_0[j]
                data[str(idx)]['aug_1'] = cur_aug_sen_1[j]
                idx = idx + 1
    with open(os.path.join(dst_path,data_type+'.json'), 'w') as outfile:
        json.dump(data, outfile,indent=4)
    return data



def preprocess(dataset='aclImdb',dst_path='../data/aclImdb/'):
    s_t = time.time()
    if os.path.exists(dst_path):
        shutil.rmtree(dst_path)
    os.mkdir(dst_path)
    s1,l1 = read_aclImdb(data_type='train')
    s1,l1,s1d,l1d = select_data(dataset,s1,l1,data_type = 'train')
    s2,l2 = read_aclImdb(data_type='test')
    s2,l2 = select_data(dataset,s2,l2,data_type = 'test')
    make_json_file(s1,l1,dst_path,data_type='train')
    make_json_file(s1d,l1d,dst_path,data_type='dev')
    make_json_file(s2,l2,dst_path,data_type='test')
    e_t = time.time()
    print(dataset,' costs ',(e_t-s_t)/3600,' hours!')
 
start_time = time.time()
if os.path.exists('../data') == False:
    os.mkdir('../data')
if os.path.exists('../data/TextClassificationDatasets') == False:
    os.mkdir('../data/TextClassificationDatasets')
preprocess('aclImdb','./data/aclImdb/')
end_time = time.time()
print('In total, it costs ',(end_time-start_time)/3600,' hours!')