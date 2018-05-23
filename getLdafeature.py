#keyword,topic interst reduce dimention with lda

import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool,Lock
import lightgbm as lgb
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import math
from sklearn.decomposition import LatentDirichletAllocation

from getPath import *
pardir = getparentdir()

from commonLib import *

contest_dir = pardir + "/preliminary_contest_data/"
construct_feature_dir = contest_dir + "/construct_features/"
raw_data_path = construct_feature_dir + '/combine_train.csv'

user_feature_path = contest_dir + '/userFeature.csv'

word_feature_dir = construct_feature_dir+'wordfeature/'
word_statis_dir = construct_feature_dir+'statis_word_feature/'
count_vector_dir = construct_feature_dir+'count_vector/'
lda_feature_dir = construct_feature_dir+'lda_feature/'
aid_lda_feature_dir = construct_feature_dir+'aid_lda_feature/'

feature_value_dir = contest_dir+'featurevalue/'

train_path = construct_feature_dir+'middle/important_train_train_20'
valid_path = construct_feature_dir+'middle/important_train_valid_20'
test_path = construct_feature_dir+'middle/important_test_20'

features = [ 'interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1', 'topic2', 'topic3']
topicsdic = {'interest1':5,'interest2':5,'interest3':3,'interest4':3,'interest5':10,'kw3':3,'kw1':5,'kw2':10,'topic1':5,'topic2':5,'topic3':3}

def get_word_feature(feature):
    user_data = pd.read_csv(user_feature_path,usecols = ['uid',feature])
    raw_data = pd.read_csv(raw_data_path)
    all_data = pd.merge(raw_data, user_data, on = 'uid', how = 'left')
    del raw_data,user_data
    all_data.to_csv(word_feature_dir + str(feature), encoding = 'utf-8',mode='w',index = False)
    del all_data
    
def get_word_feature_job(features):
    p = Pool(4)
    jobarr = []
    for feature in features:
        # lda_tune(feature)
        # get_feature_info(feature)
        job = p.apply_async(combine_aid_lda,args = (feature,))
        jobarr.append(job)
    for job in jobarr:
        job.get()
    p.close()
    p.join()
    
def handle_row(row):
    arr = []
    for r in row:
        s = str(r)
        if s == 'nan':
            continue
        arr.append(r)
        del r
    line = ','.join(arr)
    return line
    
def get_feature_info(feature):
    data = pd.read_csv(word_feature_dir+feature)
    # posdata = data[data['label'] == 1]
    # del data
    featureset = pd.DataFrame({feature+'set' :data.groupby('aid')[feature].apply(set)}).reset_index()
    del data
    sets = np.array(featureset[feature+'set'])
    # del featureset
    res = []
    for s in sets:
        s = [str(t) for t in s if not str(t)=='nan']
        line = ','.join(s)
        del s
        res.append(line)
    del sets
    # print(res)
    # vfunc = np.vectorize(handle_row)
    # print(sets)
    # res = vfunc(sets)
    featureset[feature+'set'] = res
    featureset.to_csv(word_statis_dir+feature, encoding = 'utf-8',mode='w',index = False)
    del featureset
    
def vectorize(feature):
    allfeatures = np.array(read_dic(feature_value_dir+feature))
    cv = CountVectorizer(vocabulary = allfeatures,token_pattern = '(?u)\\b\\w+\\b')
    data = pd.read_csv(word_statis_dir+feature)
    train = cv.transform(data[feature+'set'])
    write_dic(train,count_vector_dir+feature)
    
def create_lda(feature,ntopics = 1):
    ntopics = topicsdic[feature]
    tf = read_dic(count_vector_dir+feature)
    print(ntopics)
    lda = LatentDirichletAllocation(n_components=ntopics, max_iter=50,learning_method='batch',doc_topic_prior = int(50/ntopics),topic_word_prior = 0.1)
    doc_dist = lda.fit_transform(tf)
    # print(feature+":"+"topic :" +str(ntopics) +" "+str(lda.perplexity(tf,doc_dist)))
    write_dic(doc_dist,lda_feature_dir+feature)
    
def lda_tune(feature):
    for topic in [3,5,10,15]:
        create_lda(feature,topic)
        
def combine_aid_lda(feature):
    dic = np.array(read_dic(lda_feature_dir+feature))
    data = pd.read_csv(word_statis_dir+feature)
    
    df = pd.DataFrame()
    df['aid'] = data['aid']
    del data
    for i in range(topicsdic[feature]):
        df[feature+'_topic_'+str(i)] = np.round(dic[:,i],4)
    df.to_csv(aid_lda_feature_dir+feature,encoding = 'utf-8',mode='w',index = False)
        
def add_lda_feature(train_path,train_out_path):
    train_data = pd.read_csv(train_path)
    for feature in features:
        lda_data = pd.read_csv(aid_lda_feature_dir+feature)
        train_data = pd.merge(train_data,lda_data,how = 'left',on = 'aid')
        del lda_data
    train_data.to_csv(train_out_path,encoding = 'utf-8',mode='w',index = False)  

def get_cat_feature(features):
    cat = []
    for f in features:
        if '_' in f:
            continue
        cat.append(f)
        
    return cat    
    
def train():
    train_data = pd.read_csv(construct_feature_dir+'middle/train_topic')
    valid_data = pd.read_csv(construct_feature_dir+'middle/valid_topic')
    cols = train_data.columns.values
    cols = list(set(cols)-set(['aid','uid','label']))
    cat_features = get_cat_feature(cols)
    train_x = train_data[cols]
    train_y = train_data['label']
    del train_data
    
    valid_x = valid_data[cols]
    valid_y = valid_data['label']
    del valid_data
    print(cat_features)
    clf = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.05, n_estimators=500, 
    objective='binary', class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=0.8, subsample_freq=5, 
    colsample_bytree=0.8, reg_alpha=2, reg_lambda=0.001, random_state=1993, n_jobs=4)
    clf.fit(train_x,train_y,categorical_feature = cat_features, eval_set = (valid_x,valid_y),eval_metric=['auc','binary_logloss'],early_stopping_rounds = 100)
    
if __name__ == "__main__":
    # get_word_feature_job()
    get_word_feature_job(features)
    # add_lda_feature(train_path,construct_feature_dir+'middle/train_topic')
    # add_lda_feature(valid_path,construct_feature_dir+'middle/valid_topic')
    # train()


