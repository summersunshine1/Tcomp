#common function

import numpy as np
import numpy as np
import pandas as pd
import json
import pickle
import os
import re
from scipy import sparse
from sklearn.datasets import dump_svmlight_file,load_svmlight_file
from multiprocessing import Pool,Lock 
import hashlib

from getPath import *
pardir = getparentdir()

def listfiles(rootDir): 
    list_dirs = os.walk(rootDir) 
    filepath_list = []
    for root, dirs, files in list_dirs:
        for f in files:
            filepath_list.append(os.path.join(root,f))
    return filepath_list

def write_dic(dic,path):
    with open(path,'wb') as f:
        # json.dump(dic, f)
        pickle.dump(dic, f)
    
def read_dic(path):
    with open(path,'rb') as f:
        dic = pickle.load(f)
    return dic
    
def save_encoder(enc,path):
    with open(path,'wb') as f:
        dic = pickle.dump(enc, f)
        
def load_encoder(path):
    with open(path,'rb') as f:
        enc = pickle.load(f) 
    return enc

def getone_hot(feature,data):
    catearr = []
    catelist = list(data.groupby(feature).groups.keys())
    write_dic(catelist,feature)
    
def handlefetaure(f,res):
    if isinstance(f,float) or isinstance(f,int):
        res.add(str(f))
    else:
        arr = f.split(',')
        for a in arr:
            res.add(a)
    return 1

def get_all_value(feature):
    res =set()
    temp = [handlefetaure(f,res) for f in feature]
    return list(res)
    
def get_alpha_str(s):  
    result = ''.join(re.split(r'[^A-Za-z]', s))  
    pos = result.find("nan")
    if not pos == -1:
        result = result[:pos]
    return result 
    
def getfield(feature_path):
    dict = read_dic(feature_path)
    arr = list(dict.keys())
    res = [get_alpha_str(a) for a in arr]
    res = list(set(res))
    resdic = {}
    for i in range(len(res)):
        resdic[res[i]] = i+1
    return resdic
    
def normalize(df):
    df = np.round((df - df.min())/(df.max() - df.min()),5)
    return df
    
def assign(res,index,values):
    res[index] = values
    return 1
    
def convert_data_to_svm(dataarr,path):
    with open(path,mode = 'w') as f:
        f.writelines(dataarr)
        
def write_uid_aid(path,uids,aids):
    df = pd.DataFrame()
    df['uid'] = uids
    df['aid'] = aids
    # if not os.path.exists(path):
    df.to_csv(path,encoding = 'utf-8',mode='w',index = False,header = ['uid','aid'])
    # else:
        # df.to_csv(path,encoding='utf-8',mode = 'a', header=False, index = False)
        
def split(srcpath,despath):
    data = pd.read_csv(srcpath)
    ids = data.aid.unique()
    np.random.seed(1000)
    np.random.shuffle(ids)
    valsize = int(len(ids)*0.5)
    valaids = set(ids[:valsize])
    
    data['fold'] = 0
    isval = data.aid.isin(valaids)
    data.loc[isval,'fold']=1
    data.to_csv(despath,encoding='utf-8',mode = 'w', index = False)
    
def splitkfold(df,k):
    index = np.array(df.index)
    lens = len(index)
    ratio = 1.0/k
    df['fold'] = 0
    np.random.seed(1000)
    np.random.shuffle(index)
    start = 0
    for i in range(k):
        valsize = int(lens*ratio)
        valaids = set(index[start:valsize+start])
        isval = df.index.isin(valaids)
        df.loc[isval,'fold'] = i
        start = valsize+start
    return df
    
def split_fold(train_path,despath,col,k):
    print(despath)
    data = pd.read_csv(train_path)
    df = data.groupby(col).apply(splitkfold,k)
    del data
    df.to_csv(despath,encoding='utf-8',mode = 'w',index = False)
    
def save_sparse(sparse_feature,path):
    sparse.save_npz(path, sparse_feature)
    
def load_sparse(path):
    return sparse.load_npz(path)
    
def save_svm(x,y,path):
    with open(path,'wb') as f:
        dump_svmlight_file(x,y,f)
        
def load_svm(path):
    x,y = load_svmlight_file(path)
    return x,y
    
def get_file_with_nofix(file):
    return file.split('.')[0]
    
def start_job(job,dir,arg):
    files = listfiles(dir)
    p = Pool(4)
    jobarr = []
    for file in files:
        job = p.apply_async(job,args = arg)
        jobarr.append(job)
    for job in jobarr:
        job.get()
    p.close()
    p.join()

def hashstr(str1, nr_bins=1e+6):
    return int(int(hashlib.md5(str1.encode('utf8')).hexdigest(), 16)%(nr_bins-1)+1) 
    
def one_hot_representation(sample,length):
    """
    One hot presentation for every sample data
    :param fields_dict: fields value to array index
    :param sample: sample data, type of pd.series
    :param array_length: length of one-hot representation
    :return: one-hot representation, type of np.array
    """
    array = np.zeros([length])
    temp = sample.split()
    y = int(temp[0])
    # print(temp)
    for t in temp[1:]:
        featurevalue = t.split(':')
        # print(featurevalue)
        array[int(featurevalue[1])] = float(featurevalue[2])
    array = np.append(array,y)
    return array
    
def deep_one_hot(sample,length,field_count):
    array = np.zeros([length])
    temp = sample.split()
    y = int(temp[0])
    
    idx = []
    # print(temp)
    for t in temp[1:]:
        featurevalue = t.split(':')
        # print(featurevalue)
        array[int(featurevalue[1])] = float(featurevalue[2])
        idx.append(int(featurevalue[1]))
    
    res = [np.array([y]),array,np.array(idx[:field_count])]
    return res
    
def split_fold(train_path, outdir, k = 5):
    with open(train_path,'r') as f:
        col = f.readline()
        lines = f.readlines()
        l = len(lines)
        ratio = 1.0/h
        r = []
        for i in range(k+1):
            r.append(i*ratio)
        for i in range(k):
            templines = lines[int(l*r[i]):int(l*r[i+1])]
            temp = [col]
            temp += templines
            with open(outdir +str(i),'w') as ftest:
                ftest.writelines(temp)
                del temp
                
def get_cat_feature(features):
    cat = []
    for f in features:
        if '_' in f:
            continue
        cat.append(f)
        
    return cat
                
def mergefiles(train_path,valid_path, out_path):
    with open(train_path,'r') as f:
        train_lines = f.readlines()
    with open(valid_path, 'r') as f:
        f.readline()
        valid_lines = f.readlines()
    train_lines += valid_lines
    with open(out_path,'w') as f:
        f.writelines(train_lines)
  
if __name__=="__main__":
    # from getPath import *
    # pardir = getparentdir()
    # contest_dir = pardir + "/preliminary_contest_data/"
    # data = pd.read_csv(contest_dir+'/train.csv')
    # df1=pd.DataFrame({'key':['a','b','b'],'data1':range(3)})  
  
    # df2=pd.DataFrame({'key':['c','d','e'],'data1':[4,5,6]})
    # res = pd.merge(df1,df2,on=['key','data1'],how='left')
    # print(res)
    # print(data.uid.value_counts())
    # get_split_aid(contest_dir+'/train.csv',5)
    contest_dir = pardir + "/preliminary_contest_data/"
    mergefiles(contest_dir+'construct_features/middle/train_topic', contest_dir+'construct_features/middle/valid_topic', contest_dir+'construct_features/middle/all_topic_data')