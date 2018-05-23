#lightgbm model train and test

import lightgbm as lgb
import re
import pickle
import operator
from sklearn.metrics import roc_auc_score

from getPath import *
pardir = getparentdir()
from commonLib import *

contest_dir = pardir + "/preliminary_contest_data"

def get_feature_importance():
    feature_index_dic_path = contest_dir+'/ffm/featur_index_dic_1'
    # user_feature=['LBS','age','carrier','consumptionAbility','education','gender','house']
    # ad_feature = ['aid','advertiserId','campaignId','creativeId','adCategoryId', 'productId', 'productType']
    features = ['age_aid','age_campaignId','age_productType','age_productId','age_adCategoryId',
    'gender_adCategoryId','gender_advertiserId','gender_productType','gender_productId','gender_campaignId','gender_aid',
    'consumptionAbility_aid','consumptionAbility_productType','consumptionAbility_productId','education_productId','education_advertiserId',
    'carrier_campaignId']
    dic = read_dic(feature_index_path)
    l = len(dic)
    resdic = {}
    for k,v in dic.items():
        resdic[v] = k
    i = 0
    for feature in features:
        resdic[l+i] = feature
        i+=1
        
    # topic = pd.read_csv(contest_dir+'/sparse/topic_train/0.csv')
    # cols = feature_data.columns.values
    # cols = list(set(cols)-set(['aid']))
    # for col in cols:
        # resdic[l+i] = col
        # i+=1
    return resdic

def train(train_path,valid_path,model_path):
    train_data = lgb.Dataset(train_path)
    valid_data = lgb.Dataset(valid_path)
    param = {'objective':'binary','task': 'train','boosting_type':'gbdt', 'num_leaves':31,'metric': {'auc','binary_logloss'},
     'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'lambda_l1':2,
    'bagging_seed':1993
    }
    bst = lgb.train(param, train_data, num_boost_round=2000,valid_sets=[valid_data],early_stopping_rounds = 200)
    arr = bst.feature_importance()
    names = bst.feature_name()
    dictionary = dict(zip(names, arr))
    res = sorted(dictionary.items(),key=lambda x: x[1])
    print(res[-50:])
    bst.save_model(model_path)
    
def test(model_path, test_dir, res_path):
    bst = lgb.Booster(model_file=model_path)
    preds = []
    for i in range(12):
        test_path = test_dir + str(i)
        # test_data = lgb.Dataset(test_path)
        pred = list(bst.predict(test_path))
        
        preds += pred
        # del test_data
        
    write_dic(preds, res_path)
    
    
if __name__=="__main__":
    # train_path = pardir + "/preliminary_contest_data/gbm/topic_train"
    # valid_path = pardir + "/preliminary_contest_data/gbm/topic_valid"
    test_dir = pardir + "/preliminary_contest_data/sparse/topic_label_test/"
    
    # model_path = pardir + "/preliminary_contest_data/gbm_model/topic_label_model"
    res_path = pardir + "/preliminary_contest_data/gbm/res/topic_label_res"
    train_path = pardir + "/preliminary_contest_data/gbm/topic_train"
    valid_path = pardir + "/preliminary_contest_data/gbm/topic_valid"
    model_path = pardir + "/preliminary_contest_data/gbm_model/topic_label_model"
    train(train_path,valid_path,model_path)
    # test(model_path, test_dir, res_path)