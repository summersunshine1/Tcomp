#get cross feature,containing label without time will make leakage

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
user_dir = contest_dir+'/user_feature/'
user_train_dir = user_dir+'/train/'
user_test_dir = user_dir+'/test/'

combine_user_train = user_dir+'train_user.csv'
combine_user_test = user_dir+'test_user.csv'

combin_aid_fold = user_dir+'aid_fold.csv'
combin_uid_fold = user_dir+'uid_fold.csv'

add_user_train_path = user_dir+'add_train_user.csv'
add_user_test_path = user_dir +'add_test_user.csv'

train_path = contest_dir+'/train.csv'
test_path = contest_dir+'/test1.csv'


def get_info(train_path, test_path, feature, is_user, k):
    data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    m = np.round((data.label==1).mean(),4)
    c = 12
    if is_user:
        col = 'aid'
        restcol = 'uid'
    else:
        col = 'uid'
        restcol = 'aid'
    all_columns = [col,restcol,feature,feature+'ratio']
    res = pd.DataFrame(columns =all_columns,index = data.index)
    print(res.index)
    print(len(data))
    l = 0
    for i in range(k):
        data_train_0 = data[data.fold == i].reset_index(drop=1)
        data_train_1 = data[data.fold != i].reset_index(drop=1)
        l+=len(data_train_0)
        m1 = np.round((data_train_1.label==1).mean(),4)
        poscount =  pd.DataFrame({'countpos':data_train_1[data_train_1.label == 1].groupby([col,feature]).size()}).reset_index()
        count = pd.DataFrame({'count':data_train_1.groupby([col,feature]).size()}).reset_index()
        del data_train_1
        tempres = pd.merge(count,poscount,on = [col,feature],how = 'outer')
        del poscount,count
        tempres[feature+'ratio'] = round(tempres['countpos']/tempres['count'],4)
        print(tempres.head())
        tempres.drop(columns = ['countpos','count'],inplace = True)
        train_0 = pd.merge(data_train_0,tempres,on=[col,feature],how = 'left')
        print(train_0.head())
        del tempres,data_train_0
        train_0[feature+'ratio'].fillna(m1,inplace = True)
        res[data.fold == i] = np.array(train_0[all_columns])
        del train_0
    print(res.isnull().sum())
    res[col].astype('int',copy=True)
    res[restcol].astype('int',copy=True)
    poscount =  pd.DataFrame({'countpos':data[data.label == 1].groupby([col,feature]).size()}).reset_index()
    count = pd.DataFrame({'count':data.groupby([col,feature]).size()}).reset_index()
    testres = pd.merge(poscount,count,on = [col,feature])
    del poscount,count
    testres[feature+'ratio'] = round(testres['countpos']/testres['count'],4)
    testres.drop(columns = ['countpos','count'],inplace = True)
    test = pd.merge(test_data,testres,on=[col,feature],how = 'left')
    del testres,test_data
    test[feature+'ratio'].fillna(m,inplace = True)
    print(test.isnull().sum())

    res.to_csv(user_train_dir+feature+'.csv',encoding='utf-8',mode = 'w', index = False)
    
    testres = test[all_columns]
    del test
    testres.to_csv(user_test_dir+feature+'.csv',encoding='utf-8',mode = 'w', index = False)

def getcross(isuser):
    user_features = ['age','gender','education','consumptionAbility']
    ad_features = ['campaignId','adCategoryId','productId','advertiserId']

    if isuser:
        features = user_features
        folder = combin_aid_fold
    else:
        features = ad_features
        folder = combin_uid_fold
    features = ['age']
    for uf in features:
        get_info(combin_aid_fold,combine_user_test,uf,isuser,5)
    # p = Pool(4)
    # jobarr = []
    # features = ['education']
    # for uf in features:
        # job = p.apply_async(get_info,args = (combin_aid_fold,combine_user_test,uf,isuser,5,))
        # jobarr.append(job)
    # for job in jobarr:
        # job.get()
    # p.close()
    # p.join()
    
def get_statis(train_path, test_path, feature):
    data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    m = np.round((data.label==1).mean(),4)
    all_columns = ['aid',feature,feature+'ratio']
    col = 'aid'
    poscount =  pd.DataFrame({'countpos':data[data.label == 1].groupby([col,feature]).size()}).reset_index()
    count = pd.DataFrame({'count':data.groupby([col,feature]).size()}).reset_index()
    testres = pd.merge(poscount,count,on = [col,feature])
    del poscount,count
    testres[feature+'ratio'] = round(testres['countpos']/testres['count'],4)
    testres.drop(columns = ['countpos','count'],inplace = True)
    
    train = pd.merge(data,testres,on=[col,feature],how = 'left')
    del data
    train[feature+'ratio'].fillna(m,inplace = True)
    grouped = pd.DataFrame({feature+'ratio':train.groupby(['aid',feature])[feature+'ratio'].mean()}).reset_index()
    grouped[feature+'ratio'] = np.round(grouped[feature+'ratio'],4)
    del train
    grouped.to_csv(user_dir+'/sametrain/'+feature,encoding='utf-8',mode='w',index=False) 
    del grouped
    
    test = pd.merge(test_data,testres,on=[col,feature],how = 'left')
    del test_data,testres
    test[feature+'ratio'].fillna(m,inplace = True)
    grouped = pd.DataFrame({feature+'ratio':test.groupby(['aid',feature])[feature+'ratio'].mean()}).reset_index()
    grouped[feature+'ratio'] = np.round(grouped[feature+'ratio'],4)
    del test
    grouped.to_csv(user_dir+'/sametest/'+feature,encoding='utf-8',mode='w',index=False) 
    del grouped
    
def get_statis_job():
    p = Pool(4)
    jobarr = []
    features = ['age','gender','education','consumptionAbility']
    for uf in features:
        job = p.apply_async(get_statis,args = (combine_user_train,combine_user_test,uf,))
        jobarr.append(job)
    for job in jobarr:
        job.get()
    p.close()
    p.join()
 
def combine_all_feature(istrain,path):
    user_features = ['age','gender','education','consumptionAbility']
    if istrain:
        datadir = user_train_dir
    else:
        datadir = user_test_dir
    res = []
    for feature in user_features:
        data = pd.read_csv(datadir+feature+'.csv')
        if len(res)==0:
            res = data
        else:
            res[feature+'ratio'] = data[feature+'ratio']
            res[feature] = data[feature]
    res.to_csv(path,encoding='utf-8',mode='w',index=False)

def combine_diff_features(istrain):
    user_features = ['age','gender','education','consumptionAbility']
    if istrain:
        datadir = user_train_dir
        path = user_dir+'/combinetrain/'
    else:
        datadir = user_test_dir
        path = user_dir+'/combinetest/'
    for feature in user_features:
        data = pd.read_csv(datadir+feature+'.csv')
        grouped = pd.DataFrame({feature+'ratio':data.groupby(['aid',feature])[feature+'ratio'].mean()}).reset_index()
        grouped[feature+'ratio'] = np.round(grouped[feature+'ratio'],4)
        grouped[['aid',feature]] = grouped[['aid',feature]].astype('int',copy=True)
        print(grouped.head())
        grouped.to_csv(path+feature,encoding='utf-8',mode='w',index=False) 
        del grouped
        
def getcombine_data(istrain,outpath):
    user_features = ['age','gender','education','consumptionAbility']
    
    if istrain:
        datadir = user_dir+'/sametrain/'
        path = combine_user_train
    else:
        datadir = user_dir+'/sametest/'
        path = combine_user_test
    train_data = pd.read_csv(path)
    print(train_data.columns.values)
    if istrain:
        cols = ['aid','uid','label'] + user_features
    else:
        cols = ['aid','uid'] + user_features
    data = train_data[cols]
    del train_data
    
    for feature in user_features:
        feature_data = pd.read_csv(datadir+feature)
        data = pd.merge(data,feature_data,on = ['aid',feature],how ='left')
        del feature_data
    data.to_csv(outpath,encoding='utf-8',mode='w',index=False)   
    
def get_count_feature(train_path,test_path,out_dir):
    train_data = pd.read_csv(train_path)
    columns = train_data.columns.values
    columns = list(set(columns)-set(['label']))
    data = train_data[columns]
    del train_data
    test_data = pd.read_csv(test_path) 
    res = data.append(test_data)
    del test_data,data
    
    user_features = ['age','gender','education','consumptionAbility']
    for feature in user_features:
        print(feature)
        temp = pd.DataFrame({feature+'count':res.groupby(['aid',feature]).size()}).reset_index()
        temp.to_csv(out_dir+feature,encoding='utf-8',mode='w',index=False) 

    aid_features = ['campaignId','adCategoryId','productId','advertiserId']
    for feature in aid_features:
        print(feature)
        temp = pd.DataFrame({feature+'count':res.groupby(['uid',feature]).size()}).reset_index()
        temp.to_csv(out_dir+feature,encoding='utf-8',mode='w',index=False) 

if __name__=="__main__":
    # getcross(combine_user_train,True) 
    # getcross(combine_user_test,False) 
    # split_fold(combine_user_train,combin_uid_fold,'uid',5)
    # getcross(True)
    # combine_all_feature(True,add_user_train_path)
    # combine_all_feature(False,add_user_test_path)
    # combine_diff_features(True)
    # combine_diff_features(False)
    # getcombine_data(True,user_dir+'label_train_ratio')
    # getcombine_data(False,user_dir+'label_test_ratio')
    # get_statis_job()
    get_count_feature(combine_user_train,combine_user_test,user_dir+'count/')