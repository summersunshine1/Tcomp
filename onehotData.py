#construct onehot features using sklearn 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from multiprocessing import Pool,Lock
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse
from sklearn.datasets import dump_svmlight_file,load_svmlight_file

from getPath import *
pardir = getparentdir()
from commonLib import *

contest_dir = pardir + "/preliminary_contest_data"
user_split_dir = contest_dir+'/userfeature/'
feature_value_dir = contest_dir + '/featurevalue/'
fold_data_dir = contest_dir + '/fold_train/'

sparse_train_dir = contest_dir+'/sparse_train/'
sparse_train_fold_dir = contest_dir+'/sparsefold/train0/'

sparse_test_dir = contest_dir+'/sparse_test/'
sparse_test_phase2_dir = contest_dir+'/sparse_test_phase2/'

sparse_uid_dir = contest_dir+'/sparse_train_uid/'
sparse_test_uid_dir = contest_dir+'/sparse_test_uid/'
sparse_test_uid_phase2_dir = contest_dir+'/sparse_test_uid_phase2/'

onehot_model_dir = contest_dir + '/onehot/'

train_path = contest_dir+'/train.csv'
test_path = contest_dir+'/test2.csv'
adfeature_path = contest_dir + "/adFeature.csv"

user_feature_dir = contest_dir+'/user_feature/'
construct_feature_dir = contest_dir+'/construct_features/'
count_feature_dir = construct_feature_dir+'/count_feature/'

one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','advertiserId','campaignId','creativeId',
           'adCategoryId', 'productId', 'productType']
vector_feature=['os','ct','marriageStatus','appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1',
'kw2','kw3','topic1','topic2','topic3']

def preprocess(usersplit_path, raw_path,outdir,istrain):
    user_data = pd.read_csv(usersplit_path)
    train_data = pd.read_csv(raw_path)
    if istrain:
        train = train_data[['uid','aid','label']]
    else:
        train = train_data[['uid','aid']]
    del train_data
    
    # train_data.drop(columns = ['fold'],inplace = True
    res = pd.merge(train,user_data,on='uid')
    # print(usersplit_path)
    
    ad_data = pd.read_csv(adfeature_path)
    ad_data['creativeSize'] = np.round((ad_data['creativeSize'] - ad_data['creativeSize'].min())/(ad_data['creativeSize'].max() - ad_data['creativeSize'].min()),2)
    res = pd.merge(res,ad_data,on='aid')
    path = os.path.basename(usersplit_path)
    
    uids = res['uid']
    aids = res['aid']
    
    if istrain:
        write_uid_aid(sparse_uid_dir+path,uids,aids)
    else:
        write_uid_aid(sparse_test_uid_phase2_dir+path,uids,aids)
    # res.fillna('nan',inplace = True)
    train_x = res[['creativeSize']]
    if istrain:
        train_y = res['label']
    else:
        train_y = [-1]*len(train_x)
    for feature in one_hot_feature:
        print(feature)
        allfeatures = np.array(read_dic(feature_value_dir+feature))
        if 'nan' in allfeatures:
            allfeatures = list(allfeatures)
            index = allfeatures.index('nan')
            allfeatures[index] = 0
            allfeatures = np.array(allfeatures).astype(float)
            allfeatures[index] = np.max(allfeatures)+1
            res[feature].fillna(allfeatures[index],inplace = True)
        enc = OneHotEncoder()
        enc.fit(allfeatures.reshape(-1,1))
        temp = res[feature].values.reshape(-1,1)
        train = enc.transform(temp)
        train_x = sparse.hstack((train_x,train))
        
    for feature in vector_feature:
        print(feature)
        res[feature].fillna('nan',inplace = True)
        allfeatures = np.array(read_dic(feature_value_dir+feature))
        cv = CountVectorizer(vocabulary = allfeatures,token_pattern = '(?u)\\b\\w+\\b')
        train = cv.transform(res[feature])
        train_x = sparse.hstack((train_x,train))
    name = get_file_with_nofix(path)
    save_svm(train_x,train_y,outdir+name)
    

def preprocess_raw_data():
    files = listfiles(user_split_dir)
    p = Pool(4)
    jobarr = []
    for file in files:
        # job = p.apply_async(preprocess,args = (file,fold_data_dir+'train_0.csv',sparse_train_fold_dir,True,))
        # job = p.apply_async(preprocess,args = (file,train_path,sparse_train_dir,True))
        job = p.apply_async(preprocess,args = (file,test_path,sparse_test_phase2_dir,False))
        jobarr.append(job)
        # break
    for job in jobarr:
        job.get()
    p.close()
    p.join()
    # preprocess(user_split_dir+'1.csv',train_path)
    # save_encoder(enc,onehot_model_dir + "onehot")
    # save_encoder(cv,onehot_model_dir + "countvector")
 
# def add_feature(sparse_path,uid_path,ratio_path,outpath):
    # if os.path.exists(outpath):
        # return
    # print(sparse_path+" start")
    # uids = pd.read_csv(uid_path)
    # ratios = pd.read_csv(ratio_path)
    # feature_ratio = ['ageratio','genderratio','educationratio','consumptionAbilityratio']
    # res = pd.merge(uids,ratios,on = ['aid','uid'])
    # del uids,ratios
    # ratio_res = res[feature_ratio]
    # del res
    # x,y = load_svm(sparse_path)
    # x = sparse.hstack((x,ratio_res))
    
    # save_svm(x,y,outpath)
    # del x,y
    # print(sparse_path+" end")
    
# def add_feature_job(istrain):
    # if istrain:
        # sparse_dir = sparse_train_dir
        # uiddir = sparse_uid_dir 
        # ratiopath = contest_dir+'/user_feature/same_train_ratio'
        # outdir = contest_dir+'/sparse/sametrain/'
    # else:
        # sparse_dir = sparse_test_dir
        # uiddir = sparse_test_uid_dir
        # ratiopath = contest_dir+'/user_feature/same_test_ratio'
        # outdir = contest_dir+'/sparse/sametest/'
    # p = Pool(4)
    # jobarr = []
    # files = listfiles(sparse_dir)
    # for file in files:
        # base = os.path.basename(file)
        # uidpath = uiddir+base+'.csv'
        # job = p.apply_async(add_feature,args = (file,uidpath,ratiopath,outdir+base))
        # jobarr.append(job)
    # for job in jobarr:
        # job.get()
      
def add_count_feature(sparse_path, info_path, uid_path, outpath, userfeatrues,adfeatures):
    id_data = pd.read_csv(uid_path)
    info = pd.read_csv(info_path)
    merge = pd.merge(id_data,info,on=['aid','uid'],how = 'left')
    print(merge.head())
    print(str(len(id_data))+'-'+str(len(merge))+'-'+uid_path)
    del id_data,info
    
    x,y = load_svm(sparse_path)
    for feature in userfeatrues:
        print(feature)
        feature_data  = pd.read_csv(user_feature_dir+'count/' + feature)
        feature_data[feature+'count'] = normalize(feature_data[feature+'count'])
        data = pd.merge(merge,feature_data,on=['aid',feature],how = 'left')
        count = np.array(data[feature+'count']).reshape(-1,1)
        del feature_data
        x = sparse.hstack((x,count))
        del data
    for feature in adfeatures:
        feature_data  = pd.read_csv(user_feature_dir + 'count/'+feature)
        feature_data[feature+'count'] = normalize(feature_data[feature+'count'])
        data = pd.merge(merge,feature_data,on=['uid',feature],how = 'left')
        print(data.head())
        del feature_data
        x = sparse.hstack((x,data[feature+'count']))
        del data
    del merge
    save_svm(x,y,outpath)
    
def add_count_feature_job(istrain):
    if istrain:
        sparse_dir = sparse_train_dir
        uiddir = sparse_uid_dir 
        info_path  = user_feature_dir + 'train_user.csv'
        outdir = contest_dir+'/sparse/counttrain/'
        
    else:
        sparse_dir = sparse_test_dir
        uiddir = sparse_test_uid_dir
        info_path  = user_feature_dir + 'test_user.csv'
        outdir = contest_dir+'/sparse/counttest/'
        
    user_features = ['age','education','gender','consumptionAbility']
    ad_features = ['campaignId','adCategoryId','productId','advertiserId']
    p = Pool(4)
    jobarr = []
    files = listfiles(sparse_dir)
    for file in files:
        base = os.path.basename(file)
        uidpath = uiddir+base+'.csv'
        job = p.apply_async(add_count_feature,args = (file,info_path,uidpath,outdir+base,user_features, [],))
        jobarr.append(job)
    for job in jobarr:
        job.get()
        
def add_feature(sparse_path,uid_path,feature_dir,outpath):
    if os.path.exists(outpath):
        return
    print(sparse_path+" start")
    uids = pd.read_csv(uid_path)
    temparr = []
    feature_paths = listfiles(feature_dir)
    for feature_path in feature_paths:
        topic_features = pd.read_csv(feature_path)
        cols = list(set(topic_features.columns.values)-set(['aid']))
        res = pd.merge(uids,topic_features,on = ['aid'])
        del topic_features
        topic_features = res[cols]
        if len(temparr)==0:
            temparr = np.array(topic_features);
        else:
            temparr = np.hstack((temparr,topic_features))
        del res
    x,y = load_svm(sparse_path)
    x = sparse.hstack((x,temparr))
    
    save_svm(x,y,outpath)
    del x,y
    print(sparse_path+" end")
    
def add_label_to_topic(sparse_path,uid_path,label_path,outpath):
    if os.path.exists(outpath):
        return
    uids = pd.read_csv(uid_path)
    temparr = []
    label_features = pd.read_csv(label_path)
    cols = label_features.columns.values
    cats = get_cat_feature(cols)
    
    cols = list(set(cols)-set(cats))
    f_cols = cols
    f_cols += ['aid','uid']
    label_data = label_features[f_cols]
    print(f_cols)
    del label_features
    res = pd.merge(uids,label_data,on = ['aid','uid'])
    del label_data,uids
    label_features = res[cols]
    if len(temparr)==0:
        temparr = np.array(label_features);
    else:
        temparr = np.hstack((temparr,label_features))
    del res
    x,y = load_svm(sparse_path)
    x = sparse.hstack((x,temparr))
    
    save_svm(x,y,outpath)
    del x,y
    print(sparse_path+" end")
    
def add_feature_job(istrain):
    if istrain:
        sparse_dir = contest_dir + '/sparse/sparse_train/'
        uiddir = contest_dir + '/sparse/sparse_train_uid/' 
        feature_path = count_feature_dir+ 'topic_label_train.csv'
        outdir = contest_dir+'/sparse/topic_label_train/'
    else:
        sparse_dir = contest_dir + '/sparse/sparse_test/'
        uiddir = contest_dir + '/sparse/sparse_test_uid/'
        feature_path = count_feature_dir+ 'topic_label_test.csv'
        outdir = contest_dir+'/sparse/topic_label_test/'
    p = Pool(4)
    jobarr = []
    files = listfiles(sparse_dir)
    for file in files:
        base = os.path.basename(file)
        uidpath = uiddir+base+'.csv'
        # add_label_to_topic(file,uidpath,feature_path,outdir+base)
        job = p.apply_async(add_label_to_topic,args = (file,uidpath,feature_path,outdir+base))
        jobarr.append(job)
    for job in jobarr:
        job.get()

if __name__=="__main__":
    preprocess_raw_data()
    # train_ratio = contest_dir+'/user_feature/train_ratio'
    # add_feature(sparse_train_dir+'0',sparse_uid_dir+'0.csv',train_ratio)
    # add_feature_job(True)
    # add_feature_job(False)
    # add_count_feature_job(True)
        
    