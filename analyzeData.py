#split trian files into 10 parts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from multiprocessing import Pool,Lock
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import gc

from getPath import *
pardir = getparentdir()
from commonLib import *

contest_dir = pardir + "/preliminary_contest_data"
adfeature_path = contest_dir + "/adFeature.csv"
userfeature_path = contest_dir+"/userFeature.data"
userfeature_csv_path = contest_dir+"/userFeature.csv"

new_user_feature_csv_path = contest_dir+"/conv_userFeature.csv"
train_path = contest_dir+'/train.csv'
test_path = contest_dir+'/test1.csv'
sample_path = contest_dir+'/sample_train.csv'
fold_train_path = contest_dir+'/fold_train/train_0.csv'

user_split_dir = contest_dir+'/userfeature/'
new_user_split_dir = contest_dir+'/conv_userfeature/'
feature_dir = contest_dir+'/featurevalue/'

convert_feature_dir = contest_dir + '/convertdata/'
convert_raw_feature_dir = contest_dir + '/rawdata/'
convert_test_dir = contest_dir + '/converttest/'
convert_test_uid_dir = contest_dir + '/test_with_uid/'
convert_test_path = contest_dir + '/test_convert.csv'

creative_feature_dir = contest_dir + '/creativetrain/'
creative_test_dir = contest_dir + '/creativetest/'

feature_index_path = contest_dir+'/feature_index_dic'
old_feature_index_path = contest_dir+'/old_feature_index_dic'
creative_feature_index_path = contest_dir+'/cr_feature_index_dic'

user_ad_dir = contest_dir+'/user_ad_id/'
create_uid_aid_dir_train = contest_dir+'/create_ad_id_train/'
create_uid_aid_dir_test = contest_dir+'/create_ad_id_test/'
res_dir = contest_dir+'/res/'
res_path = contest_dir+'/res/testres'

field_path = contest_dir+'/field'

fold_train_dir = contest_dir + '/fold/train0/'
fold_id_train_dir = contest_dir+'/fold/id0/'

user_feature = ["uid","age","gender","marriageStatus","education","consumptionAbility","LBS","interest1","interest2","interest3",
"interest4","interest5","keyword","kw1","kw2","kw3","topic1","topic2","topic3","appIdInstall","appIdAction","ct",
"os","carrier","house"]
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']

def get_ad_info():
    data = pd.read_csv(adfeature_path)
    columns = data.columns.values;
    for c in columns:
        count = data.groupby(c).count()
        plt.title(c)
        plt.plot(count)
        plt.show()
        
def handleline(line):
    arr = line.split('|')
    dic = {}
    for a in arr:
        barr = a.split()
        if len(barr)==2:
            dic[barr[0]] = int(barr[1])
        else:
            dic[barr[0]] = ' '.join(barr[1:])
    return dic
    
def dic_to_df(dics):
    df = pd .DataFrame()
    for f in user_feature:
        values = []
        for dic in dics:
            if f in dic:
                arr = dic[f]
                arr = [str(t) for t in arr]
                ss = ','.join(arr)
                values.append(ss)
            else:
                values.append("")
        df[f] = values
    return df
    
def write_to_csv(df):
    lock.acquire()
    if not os.path.exists(new_user_feature_csv_path):
        df.to_csv(new_user_feature_csv_path,encoding = 'utf-8',mode='w',index = False)
    else:
        df.to_csv(new_user_feature_csv_path,encoding='utf-8',mode = 'a', index = False)
    lock.release()
        
def jobs(lines):
    dics = []
    for l in lines:
        dics.append(handleline(l))
    # df = dic_to_df(dics)
    userfeatures =  pd.DataFrame(dics)
    write_to_csv(userfeatures)
    
def init(l):
	global lock
	lock = l

def datatocsv():
    if os.path.exists(new_user_feature_csv_path):
        os.remove(new_user_feature_csv_path)
    lines = []
    dics = []
    lock = Lock()
    pool = Pool(4,initializer=init, initargs=(lock,))
    jobarr = []
    c = 0
    with open(userfeature_path,'r') as f:
        for cnt,line in enumerate(f):
            if len(lines)==10000:
                p = pool.apply_async(jobs ,args = (lines,))
                jobarr.append(p)
                lines = []
            line = line.strip('\n')
            lines.append(line)
    if not len(lines) == 0:
        p = pool.apply_async(jobs ,args = (lines,))
        jobarr.append(p)
    # while True:
        # time.sleep(10)
    for job in jobarr:
        job.get()
    pool.close()
    pool.join()
    
def analyzeuser():
    df = pd.read_csv(userfeature_csv_path)
    print(len(df['uid'].unique()))
    print(len(df))
    
def sampleonetrain(state = 10):
    train = pd.read_csv(train_path)
    pos = train[train['label']==1]
    neg = train[train['label']==-1]
    ratio = len(pos)*1.0/len(neg)
    print(ratio)
    # possample = pos.sample(frac=0.01,random_state = state)
    negsample = neg.sample(frac=ratio,random_state = state)
    possample = pos.append(negsample, ignore_index = False)
    possample = possample.sample(frac = 1.0,random_state = state).reset_index(drop = True)

    possample.to_csv(sample_path,encoding = 'utf-8',mode='w',index = False)
    
def chunk_to_csv(chunk,i):
    print(i)
    chunk.to_csv(new_user_split_dir+str(i)+'.csv',encoding='utf-8',mode = 'w', index = False)
    
def split_user_feature():
    chunksizes = 1000000
    data = pd.read_csv(new_user_feature_csv_path, chunksize = chunksizes)
    i = 0
    pool = Pool(4)
    jobarr = []
    for chunk in data:
        job = pool.apply_async(chunk_to_csv,args=(chunk,i,))
        jobarr.append(job)
        i+=1
    for job in jobarr:
        job.get()
    pool.close()
    pool.join()
    
def featuretodict(featurevalue,feature_name):
    values = get_all_value(featurevalue)
    write_dic(values,feature_dir+feature_name)
    
def get_ad_feature_value():
    data = pd.read_csv(adfeature_path)
    features = data.columns.values
    for feature in features:
        if feature=='aid':
            continue
        temp = data[feature]
        featuretodict(temp,feature)
        
def get_aid_feature():
    data = pd.read_csv(adfeature_path)
    temp = data['aid']
    featuretodict(temp,'aid')
    
def get_feature_value(feature):
    data = pd.read_csv(userfeature_csv_path)
    temp = data[feature]
    del data
    gc.collect()
    featuretodict(temp,feature)
    # pool = Pool(4)
    # jobarr = []
    # for f in vector_feature:
        # job = pool.apply_async(featuretodict,args=(temp[f],f,))
        # jobarr.append(job)
    # for job in jobarr:
        # job.get()
    # pool.close()
    # pool.join()  
    del temp
    gc.collect()

def mergefeature():
    user_data = pd.read_csv(user_split_dir+"0.csv")
    train_data = pd.read_csv(sample_path)
    res = pd.merge(train_data,user_data,on='uid')
    del user_data,train_data
    ad_data = pd.read_csv(adfeature_path)
    res = pd.merge(res,ad_data,on='aid')
    del ad_data
    res.fillna("nan",inplace = True)
    for f in vector_feature:
        vocab = read_dic(feature_dir+f)
        vectorizer = CountVectorizer(token_pattern = r"(?u)\b\w+\b",vocabulary = vocab)
        X = vectorizer.fit_transform(res[f])
        columns = [f+str(i) for i in range(len(vocab))]
        arr = X.toarray()
        del X
        # for i in range(len(vocab)):
            # res[columns[i]] = arr[:,i]
        t = [assign(res,columns[i],arr[:,i]) for i in range(len(vocab))]
        print(f)
    res.loc[res['label']==-1,'label']=0
    res.to_csv(convert_feature_dir+'0.csv')  
    
def getindexmap(columns,path):
    # onehot_cols = list(set(columns)-set(vector_feature))
    dict = {}
    # t = [assign(dict,onehot_cols[i],i+1) for i in range(len(onehot_cols))]
    index = 0
    for f in columns:
        vocab = read_dic(feature_dir+f)
        t = [assign(dict,f+vocab[i],index+i) for i in range(len(vocab))]
        index = index + len(vocab)
        
    dict['creativeSize'] = index
    write_dic(dict,path)
    return dict
    
# def getindexmap():
    # dict = {}
    # index = 0
    # files = listfiles(feature_dir)
    # for file in files:
        # vocab = read_dic(file)
        # f = os.path.basename(file)
        # print(f)
        # t = [assign(dict,f+vocab[i],index+i) for i in range(len(vocab))]
        # index = index + len(vocab)
    # write_dic(dict,feature_index_path)
    # return dict
        
def datarrtostr(label,arr):
    convertarr = [str(a)+":"+str(1) for a in arr]
    resarr = [str(label)]
    resarr += convertarr
    res = ' '.join(resarr)+'\n'
    return res
    
def datarrtostr_with_creativesize(label,arr,creativeSize,index):
    convertarr = [str(a)+":"+str(1) for a in arr]
    resarr = [str(label)]
    resarr += convertarr
    resarr += [str(index)+":"+str(creativeSize)]
    res = ' '.join(resarr)+'\n'
    return res
    
def handlrow(row,arr,dict,fetaure_name):
    if isinstance(row,float) or isinstance(row,int):
        arr.append(str(dict[fetaure_name+str(row)]))
    else:
        temp = row[fetaure_name].split(',')
        temp = [dict[fetaure_name+v]  for v in temp]
        arr.append(temp)

def get_csr_res(user_split_path,istrain,raw_path,feature_index_path,id_dir,des_dir):
    user_data = pd.read_csv(user_split_path)
    if istrain:
        train_data = pd.read_csv(raw_path)
    else:
        train_data = pd.read_csv(raw_path)
    train_data.drop(columns = ['fold'],inplace = True)
    res = pd.merge(train_data,user_data,on='uid')
    del user_data,train_data
    ad_data = pd.read_csv(adfeature_path)
    ad_data['creativeSize'] = np.round((ad_data['creativeSize'] - ad_data['creativeSize'].min())/(ad_data['creativeSize'].max() - ad_data['creativeSize'].min()),2)
    res = pd.merge(res,ad_data,on='aid')
    del ad_data
    res.fillna("nan",inplace = True)
    columns = set(res.columns.values)
    if istrain:
        except_cols = set(['uid','aid','label','creativeSize'])#set(['label'])
    else:
        except_cols = set(['uid','aid','creativeSize'])
    columns -= except_cols
    columns = list(columns)
    print(len(res))
    # getindexmap(columns,creative_feature_index_path)
    # return 
    dict = read_dic(feature_index_path)
    dataarr = []
    uids = res['uid']
    aids = res['aid']
    for index,row in res.iterrows():
        temparr = []
        for f in columns:
            if isinstance(row[f],float) or isinstance(row[f],int):
                temparr.append(str(dict[f+str(row[f])]))
            else:
                temp = row[f].split(',')
                temp = [dict[f+v]  for v in temp]
                temparr += temp
        dataarr.append(temparr)
    if istrain:
        label = np.array(res['label'])
    else:
        label = np.array([-1]*len(uids))
    creativesize = np.array(res['creativeSize'])
    resarr = [datarrtostr_with_creativesize(label[i],dataarr[i],creativesize[i],dict['creativeSize']) for i in range(len(label))]
    print(resarr[0])
    path = os.path.basename(user_split_path)
    if istrain:
        id_path = id_dir+path
        destpath = des_dir+path
    else:
        id_path = id_dir + path
        destpath = des_dir + path
    write_uid_aid(id_path,uids,aids)
    convert_data_to_svm(resarr,destpath)
    
def combineres():
    df = pd.DataFrame(columns=['uid','aid'])
    # files = listfiles(contest_dir +'/sparse_uid/')
    files = []
    for i in range(12):
        files.append(contest_dir +'/sparse_test_uid_phase2/'+str(i)+'.csv')
    print(files)
    # path = user_ad_dir+str(0)+".csv"
    data = pd.read_csv(files[0])
    total = len(data)
    for i in range(1,len(files)):
        # path = user_ad_dir+str(i)+".csv"
        temp = pd.read_csv(files[i])
        total += len(temp)
        data = data.append(temp,ignore_index=True)
    res = []
    with open(pardir+'/074008.csv','r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            res.append(float(line))
            
         
    # res = read_dic(pardir+'/topic_res')
    resdf = pd.DataFrame()
    res = np.round(res,8)
    
    # res = np.round_(res/(res+(1-res)*0.05),3)
    resdf['aid'] = data['aid']
    resdf['uid'] = data['uid']
    resdf['score'] = res
    # data['score'] = res
    del data
    resdf.to_csv(res_dir+'submission.csv',encoding = 'utf-8',mode='w',index = False)
    # os.system('zip submission.zip '+ resdir+'submission.csv')
    
def getfiles(dir):
    files = []
    for i in range(12):
        for j in range(10):
            temp = dir+str(i)+'_part-'+str(j)
            files.append(temp)
    return files

def combinefile():
    dir = "/apps/Difacto_DMLC/src/difacto/output/test/"
    # files = listfiles(dir)
    # files = sorted(files)
    # print(files)
    files = getfiles(dir)
    print(files)
    with open(pardir+'/combine','w') as f:
        for file in files:
            with open(file,'r') as tempfile:
                lines = tempfile.readlines()
            f.writelines(lines)
    f.close()
    


if __name__=="__main__":
    # get_ad_info()
    # datatocsv()
    # analyzeuser()
    # sampleonetrain()
    # split_user_feature()
    # mergefeature()
    # for f in vector_feature:
    # gc.collect()
    
    # cols = list(set(user_feature)-set(['uid']) - set(vector_feature))
    # for f in cols:
        # print(f)
        # get_feature_value(f)
        
    # mergefeature()
    # arr = read_dic(feature_dir+'appIdInstall')
    # arr = sorted(arr)
    # print(arr)
    
    
    # get_ad_feature_value()
    
    files = listfiles(user_split_dir)
    # for file in files:
        # get_csr_res(file,True)
    # p = Pool(4)
    # jobarr = []
    # for file in files:
        # job = p.apply_async(get_csr_res,args = (file,True,fold_train_path,creative_feature_index_path,fold_id_train_dir,fold_train_dir,))
        # jobarr.append(job)
    # for job in jobarr:
        # job.get()
    # p.close()
    # p.join()
    
    # combineres()
    # sampleonetrain()
    # get_aid_feature()
    # get_feature_value('uid')
    # getindexmap()
    # combinefile()
    combineres()
    
    