#make cross feature count statistics

import numpy as np
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool,Lock
import lightgbm as lgb
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split

from getPath import *
pardir = getparentdir()

from commonLib import *
contest_dir = pardir + "/preliminary_contest_data"
construct_feature_dir = contest_dir+'/construct_features/'
count_feature_dir = construct_feature_dir+'/count_feature/'
part_feature_path = count_feature_dir + '/partial_train.csv'
feature_dir = count_feature_dir+'/feature/'

ad_topic_feature_dir = construct_feature_dir+'/aid_lda_feature/'

user_feature=['LBS','age','carrier','consumptionAbility','education','gender','house']
ad_feature = ['aid','advertiserId','campaignId','creativeId','adCategoryId', 'productId', 'productType']

def construct_mix_feature():
    data = pd.read_csv(part_feature_path)
    for uf in user_feature:
        for af in ad_feature:
            print(uf)
            print(af)
            temp = pd.DataFrame({uf+'_'+af+'_1':data[data['label']==1].groupby([uf,af]).size()}).reset_index()
            res = pd.merge(data, temp, on = [uf,af],how = 'left')
            del temp
            res.fillna(value=0, inplace=True)
            res[uf+'_'+af+'_1'].to_csv(feature_dir+ uf+'_'+af+'_1',encoding = 'utf-8',mode='w',index = False,header = [uf+'_'+af+'_1'])
            del res
            
def construct_feature():
    data = pd.read_csv(part_feature_path)
    usefeatures = ['age']
    adfeatures = ['aid','productType']
    for uf in usefeatures:
        for af in adfeatures:
            print(uf)
            print(af)
            temp = pd.DataFrame({uf+'_'+af:data.groupby([uf,af]).size()}).reset_index()
            res = pd.merge(data, temp, on = [uf,af],how = 'left')
            del temp
            res.fillna(value=0, inplace=True)
            res[uf+'_'+af].to_csv(feature_dir+ uf+'_'+af,encoding = 'utf-8',mode='w',index = False,header = [uf+'_'+af])
            del res
            
def get_all_features():
    new_features = ['LBS','creativeId','campaignId','LBS_aid_1', 'LBS_productType_1','age_aid_1','age', 'consumptionAbility_productType_1','carrier', 
'education', 'consumptionAbility', 'LBS_creativeId_1','advertiserId','gender_aid_1','gender_advertiserId_1', 'age_aid','gender_campaignId_1',
'age_advertiserId_1','age_productType', 'consumptionAbility_creativeId_1', 'consumptionAbility_adCategoryId_1', 'age_campaignId_1',
'gender','consumptionAbility_adCategoryId', 'consumptionAbility_advertiserId_1','gender_adCategoryId_1', 'age_advertiserId',
'consumptionAbility_productType','age_campaignId','age_productType_1','house_campaignId_1','consumptionAbility_campaignId_1',
'education_productType','gender_adCategoryId','age_creativeId_1','consumptionAbility_advertiserId','gender_aid','gender_campaignId',
'gender_productType_1','age_creativeId','LBS_campaignId_1','house_adCategoryId_1','gender_creativeId','gender_productType','age_productId_1',
'age_adCategoryId_1','carrier_campaignId_1', 'consumptionAbility_campaignId','LBS_productId_1','house_aid_1', 'house_advertiserId_1',
'consumptionAbility_aid','gender_productId', 'consumptionAbility_aid_1', 'LBS_adCategoryId_1', 'gender_creativeId_1','productType','carrier_aid_1']
    new_features = new_features[:22]
    cats = get_cat_feature(new_features)
    features = list(set(new_features)-set(cats))
    data = pd.read_csv(part_feature_path)
    use_data = data[['aid','uid','label']]
    del data
    for feature in features:
        temp = pd.read_csv(feature_dir+feature)
        use_data[feature] = temp[feature]
        del temp
    
    use_data.to_csv(count_feature_dir+ 'combine_features.csv',encoding = 'utf-8',mode='w',index = False)
    
def merge_topic():
    data = pd.read_csv(part_feature_path)
    use_data = pd.DataFrame()
    merge_data = data[['aid','uid']]
    del data
    files = listfiles(ad_topic_feature_dir)
    for file in files:
        data = pd.read_csv(file)
        res = pd.merge(merge_data,data,on = 'aid',how = 'left')
        cols = res.columns.values
        cols = list(set(cols)-set(['uid','aid']))
        print(cols)
        use_data[cols] = res[cols]
        del res
    use_data.to_csv(count_feature_dir+ 'combine_topic.csv',encoding = 'utf-8',mode='w',index = False)
    
def combine_topic_label_features():
    combine_feature = pd.read_csv(count_feature_dir+ 'combine_features.csv')
    combine_feature.drop(columns=['label'],inplace =True)
    combine_topic = pd.read_csv(count_feature_dir+ 'combine_topic.csv')
    features = combine_topic.columns.values
    combine_feature[features] = combine_topic
    combine_feature.loc[0:8798814].to_csv(count_feature_dir+ 'topic_label_train.csv',encoding = 'utf-8',mode='w',index = False)
    combine_feature.loc[8798814:].to_csv(count_feature_dir+ 'topic_label_test.csv',encoding = 'utf-8',mode='w',index = False)
            
if __name__=="__main__":
    # construct_mix_feature()
    # construct_feature()
    # get_all_features()
    # merge_topic()
    combine_topic_label_features()