#using xlearn to train ffm
import xlearn as xl
from getPath import *
pardir = getparentdir()
from commonLib import *

contest_dir = pardir + "/preliminary_contest_data"
train_path = contest_dir+'/gbm/ffm_sparse_topic_train'
test_path = contest_dir+'/gbm/ffm_sparse_topic_valid'

def createffm():
    ffm_model = xl.create_ffm()
    # ffm_model.setOnDisk()
    ffm_model.setTrain(train_path)
    ffm_model.setValidate(test_path)
    param = {'lambda':0.00002, 'lr':0.05,'task':'binary','k':8,'metric':'auc'}
    ffm_model.fit(param,"./model.out")
    
if __name__=="__main__":
    createffm()
    # print(read_dic(field_path))
    
