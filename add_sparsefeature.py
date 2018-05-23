#add feature to svm format file

import os
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool,Lock
from getPath import *
pardir = getparentdir()

from commonLib import *
contest_dir = pardir + "/preliminary_contest_data/"

raw_train_dir = contest_dir+'/creativetrain/'
raw_test_dir = contest_dir+'/creativetest/'

user_dir = contest_dir+'/user_feature/'
statis_cols = ['ageratio','genderratio','educationratio','consumptionAbilityratio']
train_dir = contest_dir+"/cross_feature_train/"
test_dir = contest_dir+"/cross_feature_test/"


creative_id_train_dir = contest_dir+'/create_ad_id_train/'
creative_id_test_dir = contest_dir+'/create_ad_id_test/'

creative_feature_index_path = contest_dir+'/cr_feature_index_dic'

def handlerow(arr,index):
    temp = [str(index+i)+":"+str(round(arr[i],4)) for i in range(len(arr))]
    res = ' '.join(temp)
    return res

def combine(uid_path, start_index,istrain):
    if istrain:
        prob_data = pd.read_csv(add_user_train_path) #add feature path
        # prob_data = pd.read_csv(train_prob_add_path)
    else:
        # prob_data = pd.read_csv(test_prob_add_path)
        prob_data = pd.read_csv(add_user_test_path)
    uid_data = pd.read_csv(uid_path)
    print(prob_data.isnull().sum())
    prob_data['aid'].astype('int',copy=True)
    prob_data['uid'].astype('int',copy=True)
    data = pd.merge(uid_data,prob_data,on = ['aid','uid'])
    del uid_data,prob_data
    print(data.isnull().sum())
    # res = np.array(data[add_columns])
    res = np.array(data[statis_cols]) # add cols
    del data
    
    strres = [handlerow(res[i],start_index) for i in range(len(res))]
    return strres

def add_feature(raw_train_path,out_datadir,uid_aid_dir,istrain):
    path = os.path.basename(raw_train_path)
    uid_path = uid_aid_dir+path
    feature_dic = read_dic(creative_feature_index_path)
    index = len(feature_dic)
    featurearr = combine(uid_path, index, istrain)
    
    with open(raw_train_path,'r') as f:
        lines = f.readlines()
    reslines = []    
    for i in range(len(lines)):
        line = lines[i].strip('\n')
        line += " " + featurearr[i]+'\n'
        reslines.append(line)
    
    with open(out_datadir + path,'w') as f:
        f.writelines(reslines)
        
def start_job(istrain):
    if istrain:
        # files = listfiles(contest_dir + '/raw_add_valid/')
        files = listfiles(raw_train_dir)
        outdatadir = train_dir
        uid_aid_dir = creative_id_train_dir
    else:
        files = listfiles(raw_test_dir)
        outdatadir = test_dir
        uid_aid_dir = creative_id_test_dir
    p = Pool(4)
    jobarr = []
    # add_feature(files[0],contest_dir,uid_aid_dir)
    
    for file in files:
        job = p.apply_async(add_feature,args = (file,outdatadir,uid_aid_dir,istrain,))
        jobarr.append(job)
    for job in jobarr:
        job.get()
    p.close()
    p.join()
    
if __name__=="__main__":
    start_job(False)
    
    