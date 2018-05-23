#gbdt construct features
 
import os
import gc
import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool,Lock
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from scipy import sparse 
from sklearn import metrics
from sklearn.metrics import roc_auc_score  

from getPath import *
pardir = getparentdir()
from commonLib import *

contest_dir = pardir + "/preliminary_contest_data/"
fold_dir = contest_dir+'/fold/'
fold_train_dir0 = contest_dir+'/fold/train0/'
fold_train_id_dir0 = contest_dir+'/fold/id0/'

train_path = contest_dir+'/gbm/sparse_fold0_train'
valid_path = contest_dir+'/gbm/sparse_fold0_valid'

feature_field_index_dic_path = contest_dir+'/ffm/feature_field_dic'
feature_index_dic_path = contest_dir+'/ffm/featur_index_dic'

def testmethod():
    # param = {'objective':'binary','task': 'train','boosting_type':'gbdt', 'num_leaves':31,'metric': {'auc','binary_logloss'},
    # 'learning_rate': 0.05,
    # 'feature_fraction': 0.8,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    # 'verbose': 0,
    # 'lambda_l1':2,
    # 'bagging_seed':1993
    # }
    train_x,train_y = load_svm(train_path)
    test_x,test_y = load_svm(valid_path)
    enc = OneHotEncoder()
    leaves = 12
    nestimators = 30
    arr = np.array([i for i in range(leaves)]).reshape(-1,1)
    enc.fit(arr)
    
    # x_train1,x_train2,y_train1,y_train2 = train_test_split(train_x,train_y,test_size = 0.5,random_state = 1993)
    cl1 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=leaves, max_depth=-1, learning_rate=0.1, n_estimators=nestimators, 
    objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.8, subsample_freq=1, 
    colsample_bytree=0.8, reg_alpha=2, reg_lambda=1e-3, random_state=1993, n_jobs=4, silent=False)
    cl1.fit(train_x,train_y)#eval_metric = ['logloss','auc'],evals)
    feature_test = cl1.apply(test_x)
    # feature_train = cl1.apply(train_x)
    print(feature_test)
    return 
    # features = cl1.n_features_
    # cl2 = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=leaves, max_depth=-1, learning_rate=0.1, n_estimators=nestimators, 
    # objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.8, subsample_freq=1, 
    # colsample_bytree=0.8, reg_alpha=2, reg_lambda=1e-3, random_state=1993, n_jobs=4, silent=False)
    # cl2.fit(x_train2,y_train2)#eval_metric = ['logloss','auc'],evals)
    # temp1 = cl2.apply(x_train1)
    # print('train2 prepares')
     
    for i in range(nestimators):
        sparsetrain = enc.transform(feature_train[:,i].reshape(-1,1))
        train_x = sparse.hstack((train_x,sparsetrain))
        del sparsetrain
        
        # sparsefeature1 = enc.transform(temp1[:,i].reshape(-1,1))
        # x_train1 = sparse.hstack((x_train1,sparsefeature1))
        # del sparsefeature1
    
    # x_train = sparse.vstack((x_train1,x_train2))
    # del x_train1,x_train2
    # y_train = np.hstack((y_train1,y_train2))
    # del y_train1,y_train2
    save_svm(train_x,train_y,contest_dir+'/gbm/sparse_featureadd_train')
    del train_x,train_y
    
    # cl = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=leaves, max_depth=-1, learning_rate=0.1, n_estimators=nestimators, 
    # objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.8, subsample_freq=1, 
    # colsample_bytree=0.8, reg_alpha=2, reg_lambda=1e-3, random_state=1993, n_jobs=4, silent=False)
    # cl.fit(train_x,train_y)
    
    
    for i in range(nestimators):
        sparse_feature =  enc.transform(feature_test[:,i].reshape(-1,1))
        test_x = sparse.hstack((test_x,sparse_feature))
        del sparse_feature
    print('train prepares')  
    save_svm(test_x,test_y,contest_dir+'/gbm/sparse_featureadd_test')
    del test_x,test_y
    
def predict(isadd):
    leaves = 12
    nestimators = 30
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=leaves, max_depth=-1, learning_rate=0.1, n_estimators=nestimators, 
    objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.8, subsample_freq=1, 
    colsample_bytree=0.8, reg_alpha=2, reg_lambda=1e-3, random_state=1993, n_jobs=4, silent=False)
    if not isadd:
        train_x,train_y = load_svm(train_path)
        test_x,test_y = load_svm(valid_path)
        clf.fit(train_x,train_y)
        p = np.array(clf.predict_proba(test_x))
        print("not add feature auc:"+str(roc_auc_score(test_y, p[:,1])))

    else:
        train_x,train_y = load_svm(contest_dir+'/gbm/sparse_featureadd_train')
        test_x,test_y = load_svm(contest_dir+'/gbm/sparse_featureadd_test')
        clf.fit(train_x,train_y)
        p = np.array(clf.predict_proba(test_x))
        print("add feature auc:"+str(roc_auc_score(test_y, p[:,1])))
        

        
def get_gbdt_feature(train_path,valid_path,test_path,train_outpath, valid_outpath, test_outpath):
    # leaves = 15
    # estimator_num = 30

    train_data = pd.read_csv(train_path)
    cols = train_data.columns.values
    except_col = ['aid','uid','label']
    cols = list(set(cols)-set(except_col))
    cats = get_cat_feature(cols)
    train_x = train_data[cols]
    train_y = train_data['label']
    del train_data
    
    valid_data = pd.read_csv(valid_path)
    valid_x = valid_data[cols]
    valid_y = valid_data['label']
    del valid_data
    
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.05, n_estimators=500, 
    objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.8, subsample_freq=5, 
    colsample_bytree=0.8, reg_alpha=2, reg_lambda=0.001, random_state=1993, n_jobs=4)
    clf.fit(train_x,train_y,categorical_feature = cats, eval_set = (valid_x,valid_y),eval_metric=['auc'])
    
    # train_features = clf.apply(train_x)
    # del train_x
    # write_dic(train_features, train_outpath)
    # del train_features
    
    # valid_features = clf.apply(valid_x)
    # del valid_x
    # write_dic(valid_features, valid_outpath)
    # del valid_features
    
    # test_data = pd.read_csv(test_path)
    # test_x = test_data[cols]
    # test_features = clf.apply(test_x)
    # del test_x
    # write_dic(test_features, test_outpath)
    # del test_features
    
def start_gbdt():
    train_path = contest_dir + '/construct_features/middle/important_train_train_20'
    valid_path = contest_dir + '/construct_features/middle/important_train_valid_20'
    test_path = contest_dir + '/construct_features/partdata/important_test_20'
    
    train_out_path = contest_dir + '/construct_features/partdata/gbdtfeature/train_feature_30'
    valid_out_path = contest_dir + '/construct_features/partdata/gbdtfeature/valid_feature_30'
    test_out_path = contest_dir + '/construct_features/partdata/gbdtfeature/test_feature_30'
    get_gbdt_feature(contest_dir + '/construct_features/middle/train_topic',contest_dir + '/construct_features/middle/valid_topic',test_path,train_out_path,valid_out_path,test_out_path)
    
def handlerow(field_index, feature_index, feature, field):
    # print(feature)
    temp = [':'.join([field[i],str(feature[i]),'1']) for i in range(len(feature))]
    line = ' '.join(temp)+'\n'
    return line
    
# def handlerow(field_index, feature_index, feature):
    # print(feature)
    # return str(field_index+feature[0])+':'+str(feature_index+feature[1])+':1'
    
def add_field_and_feature_index(data_path,outpath):
    feature_index_dic = read_dic(feature_index_dic_path)
    feature_field_dic = read_dic(feature_field_index_dic_path)
    field_index = len(feature_field_dic)
    feature_index = len(feature_index_dic)

    reslines = []
    feature_data = np.array(read_dic(data_path))
    estimators = 100
    samples = len(feature_data)
    fields = np.array([i for i in range(estimators)])
    fields += field_index
    fields = np.array([str(t) for t in fields])
    
    features = feature_data + feature_index 
    del feature_data
    # vfunc = np.vectorize(handlerow)
    # reslines = vfunc(field_index,feature_index,index)
    reslines = [handlerow(field_index,feature_index, feature, fields) for feature in features]
    
    # reslines = np.fromiter((handlerow(field_index,feature_index,xi) for xi in feature_data), str, count=len(feature_data))
    print(reslines[0])
    with open(outpath,'w') as f:
        f.writelines(reslines)
    del reslines
        
def split_dic(train_path, outdir, k = 10):
    dic = read_dic(train_path)
    ratio = 1.0/k
    r = []
    for i in range(k+1):
        r.append(i*ratio)
    l = len(dic)    
    for i in range(k):
        temparr = dic[int(l*r[i]):int(l*r[i+1])]
        write_dic(temparr,outdir+"train"+str(i))
        
def get_add_feature_field_index():
    p = Pool(4)
    jobarr = []
    for i in range(10):
        # job = p.apply_async(add_field_and_feature_index,args = (contest_dir + '/construct_features/partdata/gbdtfeature/temp/train'+str(i), contest_dir + '/construct_features/partdata/gbdtfeature/tempconvert/train'+str(i),))
        # jobarr.append(job)
        add_field_and_feature_index(contest_dir + '/construct_features/partdata/gbdtfeature/temp/train'+str(i), contest_dir + '/construct_features/partdata/gbdtfeature/tempconvert/train'+str(i))
   
    # for job in jobarr:
        # job.get()
    # p.close()
    # p.join()

if __name__=="__main__":
    # testmethod()
    # predict(True)
    # predict(False)
    # start_gbdt()
    
    # add_field_and_feature_index(contest_dir + '/construct_features/partdata/gbdtfeature/test_feature_100', contest_dir + '/construct_features/partdata/gbdtfeature/ffm_test_100')
    # add_field_and_feature_index(contest_dir + '/construct_features/partdata/gbdtfeature/train_feature_100', contest_dir + '/construct_features/partdata/gbdtfeature/ffm_train_100')
    # split_fold(contest_dir + '/construct_features/partdata/gbdtfeature/train_feature_100', contest_dir + '/construct_features/partdata/gbdtfeature/temp/', 4)
    # split_dic(contest_dir + '/construct_features/partdata/gbdtfeature/train_feature_100', contest_dir + '/construct_features/partdata/gbdtfeature/temp/', k = 10)
    # get_add_feature_field_index()
    # print(len(read_dic(contest_dir + '/construct_features/partdata/gbdtfeature/test_feature_100')))
    start_gbdt()