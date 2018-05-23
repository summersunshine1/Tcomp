#fiter low frequent features

import numpy as np
import pandas as pd
import argparse, csv, sys, collections
from getPath import *
pardir = getparentdir()

from commonLib import *
contest_dir = pardir + "/preliminary_contest_data/"

train_path = contest_dir + 'train.csv'
test_path = contest_dir + 'test2.csv'
ad_path = contest_dir+'adFeature.csv'
user_split_dir = contest_dir+'/userfeature/'
adfeature_path = contest_dir + "/adFeature.csv"

feature_index_dic_path = contest_dir+'/ffm/featur_index_dic_1'
feature_field_dic_path = contest_dir+'/ffm/feature_field_dic_1'   

user_one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house']
ad_one_hot_feature = ['advertiserId','campaignId','creativeId','adCategoryId', 'productId', 'productType']
vector_feature = ['os','ct','marriageStatus','appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1',
'kw2','kw3','topic1','topic2','topic3']

def get_less_frequent_feature(threshold = 1):
    total_path = contest_dir+'total_statis'
    data = pd.read_csv(total_path)
    temp = data[data['Total']<threshold].groupby('Field')['Value'].apply(set)
    return temp

def packdata(train_path,data_path,istrain,id_dir,des_dir):
    train_data =  pd.read_csv(train_path)
    user_data = pd.read_csv(data_path)
    
    base_path = os.path.basename(data_path)
    res = pd.merge(train_data,user_data,on='uid')
    del user_data,train_data
    ad_data = pd.read_csv(adfeature_path)
    ad_data['creativeSize'] = np.round((ad_data['creativeSize'] - ad_data['creativeSize'].min())/(ad_data['creativeSize'].max() - ad_data['creativeSize'].min()),5)
    res = pd.merge(res,ad_data,on='aid')
    del ad_data
    res.fillna(-1,inplace = True)
    columns = set(res.columns.values)
    uids = res['uid']
    aids = res['aid']
    write_uid_aid(id_dir+base_path,uids,aids)
    
    dicts = res.to_dict('records')
    del res
    less_fre_features = get_less_frequent_feature()
    feature_index = read_dic(feature_index_dic_path)
    feature_field_dic = read_dic(feature_field_dic_path)
    lines = []
    for dic in dicts:
        resarr = []
        if istrain:
            label = dic['label']
            label = (1 if label==1 else 0)
        else:
            label = 0
        resarr.append(str(label))
        tempstr = str(feature_field_dic['creativeSize'])+':'+str(feature_index['creativeSize'])+':'+str(dic['creativeSize'])
        resarr.append(tempstr)
        for i in range(len(user_one_hot_feature)):
            uhot = user_one_hot_feature[i]
            if not uhot in dic:
                continue
                
            t = int(dic[uhot])
            if uhot in less_fre_features and t in less_fre_features[uhot]:
                temp = uhot+'_less'
            else:
                temp = uhot+'_'+str(t)
            tempstr = str(feature_field_dic[uhot])+':'+str(feature_index[temp])+':'+str(1)
            resarr.append(tempstr)
        for i in range(len(ad_one_hot_feature)):
            uhot = ad_one_hot_feature[i]
            if not uhot in dic:
                continue
            t = int(dic[uhot])
            if uhot in less_fre_features and t in less_fre_features[uhot]:
                temp = uhot+'_less'
            else:
                temp = uhot+'_'+str(t)
            tempstr = str(feature_field_dic[uhot])+':'+str(feature_index[temp])+':'+str(1)
            resarr.append(tempstr)
        
        for i in range(len(vector_feature)):
            uhot = vector_feature[i]
            if not uhot in dic:
                continue
            if isinstance(dic[uhot],float) or isinstance(dic[uhot],int):
                value = [int(dic[uhot])]
            else:
                value = [int(a) for a in dic[uhot].split(',')]
            for v in value:
                if uhot in less_fre_features and v in less_fre_features[uhot]:
                    temp = uhot+'_less'
                else:
                    temp = uhot+'_'+str(v)
                tempstr = str(feature_field_dic[uhot])+':'+str(feature_index[temp])+':'+str(1)
                resarr.append(tempstr)
        line = ' '.join(resarr)+'\n'
        if len(lines)==0:
            print(line)
        lines.append(line)
        
    
    with open(des_dir+base_path, 'w') as f:
        f.writelines(lines)

def packdata_job():
    files = listfiles(user_split_dir)
    p = Pool(4)
    jobarr = []
    # packdata(train_path,data_path,istrain,id_dir,des_dir)
    for file in files:
        job = p.apply_async(packdata,args = (test_path,file,False,contest_dir+'/ffm/ffm_uid_test/',contest_dir+'/ffm/ffmtest/',))
        # job = p.apply_async(packdata,args = (train_path,file,True,contest_dir+'/ffm/ffm_uid_train/',contest_dir+'/ffm/ffmtrain/',))
        jobarr.append(job)
    for job in jobarr:
        job.get()
    p.close()
    p.join()
    
def combine_test_file(dir,outpath):
    lines = []
    with open(contest_dir+'/gbm/'+outpath,'w') as ftest:
        for i in range(12):
            with open(dir+str(i)+'.csv','r') as f:
                ftest.writelines(f.readlines())
                
def combine_uid_file(dir,out_path):
    data = []
    for i in range(12):
        if len(data)==0:
            data = pd.read_csv(dir+str(i)+'.csv')
        else:
            temp = pd.read_csv(dir+str(i)+'.csv')
            data = data.append(temp)
    data.to_csv(contest_dir+'/gbm/'+out_path,encoding='utf-8',mode = 'w', index = False)
                
def combine_res():
    data = pd.read_csv(contest_dir+'/gbm/ffm_test_uid')
    res = pd.read_csv(contest_dir+'/res/0743.csv',header = None)
    finalres = pd.DataFrame()
    finalres['aid'] = data['aid']
    finalres['uid'] = data['uid']
    finalres['score'] = np.round(res,8)
    finalres.to_csv(contest_dir+'/res/submission.csv',encoding='utf-8',mode = 'w', index = False)
    
    
    
if __name__=="__main__":
    # get_less_frequent_feature(threshold = 10)
    packdata_job()
    
    # data = pd.read_csv(contest_dir+'/gbm/ffmvalid',chunksize = 100,header = None)
    # for d in data:
        # print(d)
        # break
    # del data
    # combine_uid_file(contest_dir+'/ffm/ffm_uid_test/','ffm_test_uid')
    # combine_res()