from tkinter import W
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import argparse
import numpy as np
from itertools import chain
import copy
from tqdm import tqdm
import os
import sys
from torch.utils.data import DataLoader
from utils import getCfg
from models import *
from loader import *
import pdb
import matplotlib.pyplot as plt
from train_module import *
from vit_pytorch import ViT
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

parser = argparse.ArgumentParser(description='PyTorch Biovid PreTraining')
parser.add_argument('--yamlFile', default='/hdd/sda/lzq/biovid/project/config/config.yaml', help='yaml file') 
args = parser.parse_args()

cfg=getCfg(args.yamlFile)

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["GPUID"]
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCH = cfg["EPOCH"]
SUB_EPOCH=cfg["SUB_EPOCH"]
pre_epoch = 0  
BATCH_SIZE = cfg["BATCH_SIZE"]
TCN_BATCH_SIZE=cfg["TCN_BATCH_SIZE"]
CLUSTER_EPOCH_SIZE_1=cfg["CLUSTER_EPOCH_SIZE_1"]
CLUSTER_EPOCH_SIZE_2=cfg["CLUSTER_EPOCH_SIZE_2"]
LR=cfg["LR"]
WEIGHT_DELAY=cfg["WEIGHT_DELAY"]
MODAL=cfg["MODAL"]


VGG_OR_RESNET=cfg["VGG_OR_RESNET"]
EXTRACT_NUM=cfg["EXTRACT_NUM"]
EXTRACT_NUM=5 if MODAL=='bio' else 512 if MODAL=='face' else 1
HIDDEN_NUM=cfg["HIDDEN_NUM"]
CLASS_NUM=cfg["CLASS_NUM"]

TCN_OR_LSTM=cfg["TCN_OR_LSTM"]
TCN_NUM=cfg["TCN_NUM"]
TCN_HIDDEN_NUM=cfg["TCN_HIDDEN_NUM"]

AU_INPUT_SIZE=cfg["AU_INPUT_SIZE"]
AU_HIDDEN_SIZE=cfg["AU_HIDDEN_SIZE"]
AU_OUTPUT_SIZE=cfg["AU_OUTPUT_SIZE"]

DATA_ROOT=cfg["DATA_ROOT"]
MODEL_ROOT=cfg["MODEL_ROOT"]
LOGS_ROOT=cfg["LOGS_ROOT"]
CLUSTER_ROOT=cfg["CLUSTER_ROOT"]

MODEL_NAME=cfg["MODEL_NAME"]
MODEL_NAME2=cfg["MODEL_NAME2"]
CHECKPOINT_NAME=cfg["CHECKPOINT_NAME"]
CHECKPOINT_NAME2=cfg["CHECKPOINT_NAME2"]
CHECKPOINT_NAME3=cfg["CHECKPOINT_NAME3"]
CHECKPOINT_NAME4=cfg["CHECKPOINT_NAME4"]

TRAIN_RIO=cfg["TRAIN_RIO"]
DATA_PATHS=cfg["DATA_PATHS"]
PIC_SIZE=cfg["PIC_SIZE"]
IS_POINT=cfg["IS_POINT"]

SAMPLE_THRESHOLD=cfg["SAMPLE_THRESHOLD"]
SCORE_THRESHOLD=cfg["SCORE_THRESHOLD"]
CLUSTER_NUM=cfg["CLUSTER_NUM"]

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(cfg["SEED"])

def extractor_train(modal,leave_subject=-1):
    dataset=DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=False,pic_size=PIC_SIZE,leave_subject=leave_subject)
    extractor=Resnet_classifier(modal,CLASS_NUM)
    model=SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),modal)
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor],nn.CrossEntropyLoss(),nn.L1Loss())
    #model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    if leave_subject==-1:
        model.extractor_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))
    else:
        model.extractor_train(5,os.path.join(LOGS_ROOT,CHECKPOINT_NAME+str(leave_subject)+'.t7'))

def time_extractor_train(modal,is_selfatt,is_frozen,leave_subject=-1):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,pic_size=PIC_SIZE,leave_subject=leave_subject)
    # extractor=Resnet_classifier(modal,CLASS_NUM)
    extractor=NoChange()
    # time_extractor=Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM)
    # time_extractor=TCN(EXTRACT_NUM,HIDDEN_NUM*2)
    time_extractor=ViT(image_size = PIC_SIZE,
        patch_size = 28,
        num_classes = HIDDEN_NUM*2,
        dim = EXTRACT_NUM,
        depth = 3,
        heads = 4,
        mlp_dim = HIDDEN_NUM,
        dropout = 0.1,
        emb_dropout = 0.1)
    model=SingleModel(extractor,time_extractor,Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),modal)
    # if leave_subject==-1:
    #     model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    # else:
    #     model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME+str(leave_subject)+'.t7')))
    if is_frozen:
        model.train_init(dataset,LR,WEIGHT_DELAY,[model.time_extractor,model.regressor],nn.CrossEntropyLoss(),nn.L1Loss())
    else:
        model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor,model.time_extractor,model.regressor],nn.CrossEntropyLoss(),nn.L1Loss())
    # model.load_time_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    #+str(leave_subject)+'.t7'
    if leave_subject==-1:
        model.time_extractor_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME),is_selfatt=is_selfatt,is_frozen=is_frozen)
    else:
        model.time_extractor_train(30,os.path.join(LOGS_ROOT,MODEL_NAME+str(leave_subject)+'_epoch30.t7'),is_selfatt=is_selfatt,is_frozen=is_frozen)

def extractor_test(modal):
    dataset=DataSet(BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=False,collate_fn=None,pic_size=PIC_SIZE)
    #extractor=VGG_regressor() if modal=='face' else Resnet_regressor(modal)
    model=SingleModel(Resnet_regressor(modal),TCN(512,64,TCN_HIDDEN_NUM),Regressor_self(64,64,64,is_droup=0.6),modal)
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.extractor],nn.MSELoss(),nn.L1Loss())
    print(model.extractor_test(model.test_criterion))

def voice_train():
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,'face',is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    facemodel=SingleModel(Resnet_regressor('face'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"face")
    voicemodel=SingleModel(Resnet_regressor('voice'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"voice")
    model=TwoModel(facemodel,voicemodel,Voice_Time_CrossAttention(EXTRACT_NUM,HIDDEN_NUM))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.CrossModel,model.VoiceModel.regressor])
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)))
    model.voice_train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def three_train():
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,'bio',is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    facemodel=SingleModel(Resnet_regressor('face'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"face")
    voicemodel=SingleModel(Resnet_regressor('voice'),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"voice")
    biomodel=BioModel(Time_SelfAttention(3,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),'bio')
    model=TwoModel(facemodel,voicemodel,Voice_Time_CrossAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor_self_att(3,HIDDEN_NUM/2,HIDDEN_NUM/2,0.1),biomodel)
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.regressor])
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)),bio_checkout=torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3)))
    model.train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))
    
def cluster_train(modal,is_selfatt,checkpoint,pre_model_id,model_id,pre_score,save_cluster,cluster_num,leave_subject=-1):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,pic_size=PIC_SIZE,leave_subject=leave_subject)
    # extractor=Resnet_classifier(modal,CLASS_NUM)
    extractor=NoChange()
    time_extractor=ViT(image_size = PIC_SIZE,
        patch_size = 28,
        num_classes = HIDDEN_NUM*2,
        dim = EXTRACT_NUM,
        depth = 3,
        heads = 4,
        mlp_dim = HIDDEN_NUM,
        dropout = 0.1,
        emb_dropout = 0.1)
    cluster=ClusterCenter(HIDDEN_NUM*2,cluster_num)
    model=SingleModel(extractor,time_extractor,Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),modal,cluster=cluster)
    model.load_time_checkpoint(checkpoint["modelList"][pre_model_id])
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.cluster],nn.MSELoss(),nn.L1Loss())
    return model.feature_space(is_selfatt=is_selfatt,savepath=os.path.join(save_cluster,str(pre_model_id)),pre_space_path=checkpoint['space_path'],pre_model_id=pre_model_id,model_id=model_id,sample_threshold=SAMPLE_THRESHOLD,score_threshold=SCORE_THRESHOLD,pre_score=pre_score,cluster_num=cluster_num,CLUSTER_EPOCH_SIZE_1=CLUSTER_EPOCH_SIZE_1)

def ori_MultiExperts_train(modal):
    #space_path,centerList=cluster_train(modal,True)

    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,pic_size=PIC_SIZE)
    extractor=NoChange() if modal=='bio' else Resnet_regressor(modal)
    modelList=[]
    checkList=[]
    for _ in range(3):
        modelList.append(SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal))
        checkList.append(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    experts=MultiExperts(modelList)
    experts.load_checkpoint(checkList,torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
    experts.train_init(dataset,LR,WEIGHT_DELAY)
    experts.train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME))

def MultiExperts_train(modal,is_frozen=True,leave_subject=-1):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,pic_size=PIC_SIZE,leave_subject=leave_subject)
    checkpointname=os.path.join(LOGS_ROOT,MODEL_NAME+str(leave_subject)+'_epoch30.t7')
    checkpoint=torch.load(checkpointname)
    checkpoint["modelList"]=[{"extractor":checkpoint["extractor"],"time_extractor":checkpoint["time_extractor"],"regressor":checkpoint["regressor"]}]
    checkpoint["score"]=1e3
    checkpoint['space_path']={}

    cnt=0

    while 1:
        if cnt>=1:
            break

        print("---------------------Generation:",cnt)
        space_path={}  
        modelList=[]
        checkList=[]
        save_model=os.path.join(LOGS_ROOT,MODEL_NAME+str(leave_subject)+'_epoch30_experts_'+str(cnt)+"G.t7")
        save_cluster=os.path.join(CLUSTER_ROOT,MODEL_NAME+str(leave_subject)+'_epoch30_experts_'+str(cnt)+"G.t7")
        
        for i in range(len(checkpoint["modelList"])):
            space_path_sub,num,centerList,stdList=cluster_train(modal,is_selfatt=True,checkpoint=checkpoint,pre_model_id=i,model_id=len(modelList),pre_score=checkpoint["score"],save_cluster=save_cluster,cluster_num=CLUSTER_NUM,leave_subject=leave_subject)
            space_path.update(space_path_sub)
            for _ in range(num):
                # extractor=Resnet_classifier(modal,CLASS_NUM)
                extractor=NoChange()
                time_extractor=ViT(image_size = PIC_SIZE,
                    patch_size = 28,
                    num_classes = HIDDEN_NUM*2,
                    dim = EXTRACT_NUM,
                    depth = 3,
                    heads = 4,
                    mlp_dim = HIDDEN_NUM,
                    dropout = 0.1,
                    emb_dropout = 0.1)
                modelList.append(SingleModel(extractor,time_extractor,Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),modal))
                checkList.append(checkpoint["modelList"][i])

        if len(modelList)==len(checkpoint["modelList"]):
            break
        # return
        experts=MultiExperts(modelList)
        experts.load_checkpoint(checkList,space_path=space_path,centerList=centerList,stdList=stdList)
        experts.train_init(dataset,LR,WEIGHT_DELAY,is_frozen)
        extractor=NoChange()
        time_extractor=ViT(image_size = PIC_SIZE,
            patch_size = 28,
            num_classes = HIDDEN_NUM*2,
            dim = EXTRACT_NUM,
            depth = 3,
            heads = 4,
            mlp_dim = HIDDEN_NUM,
            dropout = 0.1,
            emb_dropout = 0.1)
        experts.protype_train(backbone=SingleModel(extractor,time_extractor,Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),modal),checkpoint=torch.load(checkpointname),protype=Cluster(HIDDEN_NUM*2,HIDDEN_NUM,CLUSTER_NUM),CLUSTER_EPOCH_SIZE_2=CLUSTER_EPOCH_SIZE_2,num=CLUSTER_NUM,savepath=os.path.join(save_cluster,str(0)))
        experts.train(EPOCH,save_model,num)
        checkpoint=torch.load(save_model)
        cnt+=1

def MultiExperts_checkpoint_train(modal):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)

    checkpoint=torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME))

    modelList=[]
    checkList=[]
    save_model=os.path.join(LOGS_ROOT,CHECKPOINT_NAME)

    for i in range(len(checkpoint["modelList"])):
        extractor=NoChange() if modal=='bio' else Resnet_regressor(modal)
        modelList.append(SingleModel(extractor,Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal))
        checkList.append(checkpoint["modelList"][i])
    
    experts=MultiExperts(modelList)
    experts.load_checkpoint(checkList,space_path=checkpoint["space_path"],centerList=checkpoint["centerList"],stdList=checkpoint["stdList"])
    experts.train_init(dataset,LR,WEIGHT_DELAY)
    experts.testloss_best=checkpoint["acc"]
    print(checkpoint["acc"])
    experts.train(EPOCH,save_model)

def MultiExperts_test(modal,modelNum):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,modal,is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    modelList=[]
    for _ in range(modelNum):
        modelList.append(SingleModel(Resnet_regressor(modal),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal))
    backbone=SingleModel(Resnet_regressor(modal),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),modal)
    experts=MultiExperts(modelList,backbone)
    experts.test_init(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),dataset)
    experts.test()

def Mul_MultiExperts_test(modelNum):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,"bio",is_time=True,collate_fn=collate_fn,pic_size=PIC_SIZE)
    modelList=[]
    # for _ in range(modelNum):
    #     modelList.append(SingleModel(Resnet_regressor("face"),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"face"))
    # for _ in range(modelNum):
    #     modelList.append(SingleModel(Resnet_regressor("voice"),Time_SelfAttention(EXTRACT_NUM,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"voice"))
    for _ in range(modelNum):
        modelList.append(SingleModel(NoChange(),Time_SelfAttention(3,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"bio"))
    experts=MultiExperts(modelList)
    checkpointList=[torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3))]
    #checkpointList=[torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2))]
    #checkpointList=[torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3))]
    experts.mul_test_init(checkpointList,dataset)
    experts.test()

def Mul_MultiExperts_train(modelNum,leave_subject):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,"all",is_time=True,pic_size=PIC_SIZE,leave_subject=leave_subject)
    modelList=[]
    backboneList=[]
    extractor=NoChange()
    time_extractor=ViT(image_size = PIC_SIZE,
        patch_size = 28,
        num_classes = HIDDEN_NUM*2,
        dim = EXTRACT_NUM,
        depth = 3,
        heads = 4,
        mlp_dim = HIDDEN_NUM,
        dropout = 0.1,
        emb_dropout = 0.1)
    backboneList.append(SingleModel(Resnet_classifier("face",CLASS_NUM),Time_SelfAttention(512,64),Classifier(64*2,64,CLASS_NUM),"face"))
    backboneList.append(SingleModel(extractor,time_extractor,Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),"bio"))

    for _ in range(modelNum[0]):
        modelList.append(SingleModel(Resnet_classifier("face",CLASS_NUM),Time_SelfAttention(512,64),Classifier(64*2,64,CLASS_NUM),"face"))
    for _ in range(modelNum[1]):
        extractor=NoChange()
        time_extractor=ViT(image_size = PIC_SIZE,
            patch_size = 28,
            num_classes = HIDDEN_NUM*2,
            dim = EXTRACT_NUM,
            depth = 3,
            heads = 4,
            mlp_dim = HIDDEN_NUM,
            dropout = 0.1,
            emb_dropout = 0.1)
        modelList.append(SingleModel(extractor,time_extractor,Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),"bio"))
    # for _ in range(modelNum):
    #     modelList.append(SingleModel(NoChange(),Time_SelfAttention(3,HIDDEN_NUM),Regressor(HIDDEN_NUM*2,HIDDEN_NUM),"bio"))
    experts=MultiExperts(modelList)
    # checkpointList=[torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3))]
    checkpointList=[torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME+str(leave_subject)+'_epoch2_experts_0G.t7')),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2+str(leave_subject)+'_epoch30_experts_0G.t7'))]
    #checkpointList=[torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2)),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME3))]
    for checkpoint in checkpointList:
        show(checkpoint)
    experts.mul_train_init(checkpointList,dataset,backboneList,Classifierxplusy(128,HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),modelNum)
    # experts.regressor.load_state_dict(torch.load(os.path.join(LOGS_ROOT,MODEL_NAME2+str(leave_subject)+'_2experts.t7'))['regressor'])
    experts.train_fusioner(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME2+str(leave_subject)+'_2experts.t7'))

def two_train(leave_subject):
    dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,"all",is_time=True,pic_size=PIC_SIZE,leave_subject=leave_subject)
    extractor_voice=NoChange()
    time_extractor_voice=ViT(image_size = PIC_SIZE,
        patch_size = 28,
        num_classes = HIDDEN_NUM*2,
        dim = EXTRACT_NUM,
        depth = 3,
        heads = 4,
        mlp_dim = HIDDEN_NUM,
        dropout = 0.1,
        emb_dropout = 0.1)
    facemodel=SingleModel(Resnet_classifier("face",CLASS_NUM),Time_SelfAttention(512,64),Classifier(64*2,64,CLASS_NUM),"face")
    voicemodel=SingleModel(extractor_voice,time_extractor_voice,Classifier(HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM),"bio")
    model=TwoModel(facemodel,voicemodel,Classifierxplusy(128,HIDDEN_NUM*2,HIDDEN_NUM,CLASS_NUM))
    model.train_init(dataset,LR,WEIGHT_DELAY,[model.regressor])
    model.load_checkpoint(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME+str(leave_subject)+'_epoch10.t7')),torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2+str(leave_subject)+'_epoch30.t7')))
    model.train(EPOCH,os.path.join(LOGS_ROOT,MODEL_NAME2+str(leave_subject)+'.t7'))


def show(checkpoint):
    keys=['acc','cm']
    for key in keys:
        print(key,checkpoint[key])

def f1(cm):
    confusion_matrix = np.array(cm)*20

    # Calculate metrics
    true_positives = confusion_matrix[1, 1]
    false_positives = confusion_matrix[0, 1]
    false_negatives = confusion_matrix[1, 0]
    true_negatives = confusion_matrix[0, 0]

    accuracy = (true_positives + true_negatives) / np.sum(confusion_matrix)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Round the results to 5 decimal places
    # accuracy = round(accuracy, 5)
    # f1_score = round(f1_score, 5)

    print(accuracy,f1_score)
    return f1_score

def multi_person(start,end):
    acc=0
    f1_score=0
    for i in range(start,end):
        # extractor_train(MODAL,leave_subject=i)
        # time_extractor_train(MODAL,is_selfatt=True,is_frozen=True,leave_subject=i)
        # MultiExperts_train(MODAL,leave_subject=i)
        # two_train(i)
        # Mul_MultiExperts_train([2,2],i)
        checkpoint=torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME2+str(i)+'_epoch30_experts_0G.t7'))
        acc+=checkpoint['acc']
        f1_score+=f1(checkpoint['cm'])

    print(acc/(end-start),f1_score/(end-start))

# show(torch.load(os.path.join(LOGS_ROOT,CHECKPOINT_NAME)))
# show(torch.load("/hdd/sda/lzq/biovid/model/logs/face_0vs3_gen_76.t7"))
# extractor_train(MODAL)
# multi_person(74,81)
# time_extractor_train(MODAL,is_selfatt=True,is_frozen=True)
# cluster_train(MODAL,is_selfatt=True)
# for i in range(74,81):
#     MultiExperts_train(MODAL,leave_subject=i)
#MultiExperts_test(MODAL,3)
#Mul_MultiExperts_test(3)
#MultiExperts_checkpoint_train(MODAL)

# Mul_MultiExperts_train([2,2],74)

from scipy.signal import find_peaks
import numpy as np
# from entropy import app_entropy, sample_entropy
from scipy.stats import linregress, skew, kurtosis

def detect_qrs_complexes(ecg_signal, fs):
    # Differentiate the signal
    diff_ecg = np.diff(ecg_signal)

    # Square the differentiated signal
    squared_ecg = diff_ecg ** 2

    # Moving-average filter
    window_size = int(0.08 * fs)  # 80 ms window size
    ma_ecg = np.convolve(squared_ecg, np.ones(window_size) / window_size, mode='same')

    # Find QRS complex peaks
    qrs_peaks, _ = find_peaks(ma_ecg, distance=int(0.2 * fs), height=0.4 * np.max(ma_ecg))

    return qrs_peaks

def biophys_feature_extraction(data):
    emg_gsr_list=[1,3,4,5]
    features=[]
    for i in emg_gsr_list:
        filtered_emg=data['xs'][i].squeeze().numpy()
        peak_height = np.max(filtered_emg)
        peak_difference = np.max(filtered_emg) - np.min(filtered_emg)
        mean_absolute_difference = np.mean(np.abs(np.diff(filtered_emg)))
        # fourier_coefficients = np.fft.fft(filtered_emg)
        # bandwidth = np.std(filtered_emg)
        # approximate_entropy = app_entropy(filtered_emg, order=2, metric='chebyshev')
        # sample_entropy = sample_entropy(filtered_emg, order=2, metric='chebyshev')
        stationarity = linregress(np.arange(len(filtered_emg)), filtered_emg).pvalue
        statistical_moments = [np.mean(filtered_emg), np.std(filtered_emg), np.var(filtered_emg),
                            skew(filtered_emg), kurtosis(filtered_emg)]
        feature=[peak_height,peak_difference,mean_absolute_difference,stationarity]+statistical_moments
        for fea in feature:
            features.append(fea)

    filtered_ecg=data['xs'][2].squeeze().numpy()
    # Detect QRS complexes (you may need to use a QRS detection algorithm, e.g., Pan-Tompkins)
    qrs_complexes = detect_qrs_complexes(filtered_ecg, 256)
    if len(qrs_complexes)<2:
        return None

    # Compute RR intervals
    rr_intervals = np.diff(qrs_complexes)

    # Compute features for ECG signal
    mean_difference = np.mean(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    slope, _, _, _, _ = linregress(np.arange(len(rr_intervals)), rr_intervals)

    feature=[mean_difference, rmssd, slope]
    for fea in feature:
        features.append(fea)
    return np.array(features)


def xy_bio_np(dataloader):
    x_list=[]
    y_list=[]
    bar = tqdm(total=len(dataloader))
    for data in dataloader:
        feature=biophys_feature_extraction(data)
        if feature is None:
            continue
        x_list.append(feature)
        y_list.append(int(data['y']))
        bar.update(1)
    bar.close()
    x_np = np.array(x_list)
    y_np = np.array(y_list)

    return x_np,y_np

def visual_svm(clf,x_train,y_train,leave_subject):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    # Project data onto first two t-SNE components
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_train)

    # Create grid to evaluate model
    xx = np.linspace(np.min(x_tsne[:, 0]), np.max(x_tsne[:, 0]), 30)
    yy = np.linspace(np.min(x_tsne[:, 1]), np.max(x_tsne[:, 1]), 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # Compute decision function for each point in the grid
    Z = np.zeros_like(XX)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            # Find nearest training point to current grid point
            dists = np.sum((x_tsne - xy[i * Z.shape[1] + j])**2, axis=1)
            nearest_idx = np.argmin(dists)

            # Evaluate decision function at nearest training point
            Z[i, j] = clf.decision_function([x_train[nearest_idx]])

    # Plot decision boundary and margins
    plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    # # Plot support vectors
    # plt.scatter(x_tsne[clf.support_, 0], x_tsne[clf.support_, 1], s=100,
    #             linewidth=1, facecolors='none', edgecolors='k')

    # Plot data points
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y_train, s=10,cmap='coolwarm')

    plt.savefig('/hdd/sda/lzq/biovid/dataset/dataset/bio_svm_'+str(leave_subject)+'.jpg')
    
    plt.close()

def reg_method(reg,x_train,y_train,x_test,i):
    if reg=='inv':
        x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])

        x_pinv = np.linalg.pinv(x_train)

        coefficients = x_pinv @ y_train

        x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
        y_pred_prob=x_test @ coefficients
        y_pred = (x_test @ coefficients>0.5).astype(int)
        

    else:
        clf=SVC(kernel='poly') if reg=='svr' else RandomForestClassifier()
        clf.fit(x_train, y_train)
        y_pred_prob = clf.predict_proba(x_test)[:, 1]
        y_pred = clf.predict(x_test)
        if reg=='svr':
            visual_svm(clf,x_train,y_train,i)

    return y_pred,y_pred_prob

def time_process(fea):
    x_np=fea.numpy()
    from scipy import signal,stats

    # Apply Butterworth low-pass filter to the signals
    b, a = signal.butter(5, 1, 'low', fs=x_np.shape[0])
    x_filtered = signal.filtfilt(b, a, x_np, axis=0)

    # Estimate the first and second temporal derivatives of the signals
    dx = np.gradient(x_filtered, axis=0)
    d2x = np.gradient(dx, axis=0)

    # Compute statistical measures for each signal and its derivatives
    res = np.zeros((21*x_np.shape[1]))
    for i in range(x_np.shape[1]):
        res[i*21:(i+1)*21] = [np.mean(x_filtered[:, i]), np.median(x_filtered[:, i]), np.ptp(x_filtered[:, i]), np.std(x_filtered[:, i]),
                                stats.median_abs_deviation(x_filtered[:, i]), np.subtract(*np.percentile(x_filtered[:, i], [75, 25])),
                                np.subtract(*np.percentile(x_filtered[:, i], [90, 10])),
                                np.mean(dx[:, i]), np.median(dx[:, i]), np.ptp(dx[:, i]), np.std(dx[:, i]),
                                stats.median_abs_deviation(dx[:, i]), np.subtract(*np.percentile(dx[:, i], [75, 25])),
                                np.subtract(*np.percentile(dx[:, i], [90, 10])),
                                np.mean(d2x[:, i]), np.median(d2x[:, i]), np.ptp(d2x[:, i]), np.std(d2x[:, i]),
                                stats.median_abs_deviation(d2x[:, i]), np.subtract(*np.percentile(d2x[:, i], [75, 25])),
                                np.subtract(*np.percentile(d2x[:, i], [90, 10]))]
    return res

def xy_face_np(dataloader):
    x_list=[]
    y_list=[]
    bar = tqdm(total=len(dataloader))
    for data in dataloader:
        nvs1=[]
        nvs2=[]
        aus=[]
        gra=[]
        face_points=data['xs'][7]
        for face_point in face_points:
            # if not (face_point[2][0] == -1).any():
            nvs1.append(face_point[0][0][0])
            nvs2.append(face_point[0][0][1])
            aus.append(face_point[1][0])
            # gra.append(face_point[2][0])
        if len(aus)<20:
            continue
        fea=[nvs1,nvs2,aus]
        for i in range(len(fea)):
            fea[i]=torch.stack(fea[i])
            fea[i]=fea[i]-fea[i].mean(axis=0)
            # fea[i]=fea[i]-fea[i].mean(axis=0) if i<3 else fea[i]/fea[i].mean(axis=0)
        fea=torch.cat(fea,dim=1)
        # fea=(fea-fea.mean(axis=0)+3*fea.std(axis=0))/(6*fea.std(axis=0))
        x_list.append(time_process(fea))
        y_list.append(int(data['y']))
        bar.update(1)
    bar.close()
    x_np = np.array(x_list)
    y_np = np.array(y_list)

    return x_np,y_np

def lbp_top(video, radius=3, n_points=8):
    from skimage import feature
    """
    Compute LBP-TOP features for an RGB video sequence.
    
    Parameters
    ----------
    video : ndarray
        Input RGB video sequence with shape (n_frames, height, width, 3).
    radius : int
        Radius of the circular LBP pattern.
    n_points : int
        Number of points in the circular LBP pattern.
    
    Returns
    -------
    hist : ndarray
        Histogram of LBP-TOP features with shape (n_bins,).
    """
    video=np.transpose(video*255,(0,2,3,1))
    n_frames, height, width, _ = video.shape
    
    # Convert RGB video to grayscale
    gray_video = np.mean(video, axis=3)
    
    # Compute LBP for XY plane
    lbp_xy = np.zeros((n_frames, height, width), dtype=np.uint8)
    for i in range(n_frames):
        lbp_xy[i] = feature.local_binary_pattern(gray_video[i], n_points,
                                                 radius, method='uniform')
    
    # Compute LBP for XT plane
    lbp_xt = np.zeros((height, n_frames, width), dtype=np.uint8)
    for i in range(height):
        lbp_xt[i] = feature.local_binary_pattern(gray_video[:, i], n_points,
                                                 radius, method='uniform')
    
    # Compute LBP for YT plane
    lbp_yt = np.zeros((width, n_frames, height), dtype=np.uint8)
    for i in range(width):
        lbp_yt[i] = feature.local_binary_pattern(gray_video[:, :, i], n_points,
                                                 radius, method='uniform')
    
    # Concatenate histograms of all three planes
    hist_xy = np.histogram(lbp_xy.ravel(), bins=n_points+2,
                           range=(0, n_points+2))[0]
    hist_xt = np.histogram(lbp_xt.ravel(), bins=n_points+2,
                           range=(0, n_points+2))[0]
    hist_yt = np.histogram(lbp_yt.ravel(), bins=n_points+2,
                           range=(0, n_points+2))[0]
    
        # Calculate the total number of pixels in the image
    total_pixels = lbp_xy.size

    # Calculate the percentage of pixels that fall within each bin
    percent_xy = hist_xy / total_pixels
    percent_xt = hist_xt / total_pixels
    percent_yt = hist_yt / total_pixels
    
    hist = np.concatenate((percent_xy, percent_xt, percent_yt))
    
    return hist

def xy_frame_np(dataloader):
    x_list=[]
    y_list=[]
    bar = tqdm(total=len(dataloader))
    for data in dataloader:
        imgs=data['xs'][0][0].numpy()
        lbp_top_feature=lbp_top(imgs)
        x_list.append(lbp_top_feature)
        y_list.append(int(data['y']))
        bar.update(1)
    bar.close()
    x_np = np.array(x_list)
    y_np = np.array(y_list)

    return x_np,y_np
    
def xy_frame_bio_np(dataloader):
    x_list=[]
    y_list=[]
    bar = tqdm(total=len(dataloader))
    for data in dataloader:
        imgs=data['xs'][0][0].numpy()
        lbp_top_feature=lbp_top(imgs)
        bio_feature=biophys_feature_extraction(data)
        if bio_feature is None:
            continue
        x_list.append(np.concatenate((lbp_top_feature,bio_feature),axis=-1))
        y_list.append(int(data['y']))
        bar.update(1)
    bar.close()
    x_np = np.array(x_list)
    y_np = np.array(y_list)

    return x_np,y_np

def xy_face_bio_np(dataloader):
    x_list=[]
    y_list=[]
    bar = tqdm(total=len(dataloader))
    for data in dataloader:
        nvs1=[]
        nvs2=[]
        aus=[]
        gra=[]
        face_points=data['xs'][7]
        for face_point in face_points:
            # if not (face_point[2][0] == -1).any():
            nvs1.append(face_point[0][0][0])
            nvs2.append(face_point[0][0][1])
            aus.append(face_point[1][0])
            # gra.append(face_point[2][0])
        if len(aus)<20:
            continue
        fea=[nvs1,nvs2,aus]
        for i in range(len(fea)):
            fea[i]=torch.stack(fea[i])
            fea[i]=fea[i]-fea[i].mean(axis=0)
            # fea[i]=fea[i]-fea[i].mean(axis=0) if i<3 else fea[i]/fea[i].mean(axis=0)
        fea=torch.cat(fea,dim=1)
        fea=time_process(fea)
        
        bio_feature=biophys_feature_extraction(data)
        if bio_feature is None:
            continue
        x_list.append(np.concatenate((fea,bio_feature),axis=-1))
        y_list.append(int(data['y']))
        bar.update(1)
    bar.close()
    x_np = np.array(x_list)
    y_np = np.array(y_list)

    return x_np,y_np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
def bio_pre(start,end):
    accs=[0,0]
    # pres=[0,0,0]
    # recs=[0,0,0]
    f1s=[0,0]
    aucs=[0,0]
    matrics=[accs,f1s,aucs]
    for i in range(start,end):
        if os.path.exists('/hdd/sda/lzq/biovid/dataset/dataset/y_face_bio_train_'+str(i)+'.npy'):
            x_test=np.load('/hdd/sda/lzq/biovid/dataset/dataset/x_face_bio_test_'+str(i)+'.npy')
            y_test=np.load('/hdd/sda/lzq/biovid/dataset/dataset/y_face_bio_test_'+str(i)+'.npy')
            x_train=np.load('/hdd/sda/lzq/biovid/dataset/dataset/x_face_bio_train_'+str(i)+'.npy')
            y_train=np.load('/hdd/sda/lzq/biovid/dataset/dataset/y_face_bio_train_'+str(i)+'.npy')
        else:
            dataset=DataSet(TCN_BATCH_SIZE,TRAIN_RIO,DATA_PATHS,"all",is_time=True,pic_size=PIC_SIZE,leave_subject=i)
            x_test,y_test=xy_face_bio_np(dataset.test_dataloader)
            np.save('/hdd/sda/lzq/biovid/dataset/dataset/x_face_bio_test_'+str(i)+'.npy',x_test)
            np.save('/hdd/sda/lzq/biovid/dataset/dataset/y_face_bio_test_'+str(i)+'.npy',y_test)
            x_train,y_train=xy_face_bio_np(dataset.train_dataloader)
            np.save('/hdd/sda/lzq/biovid/dataset/dataset/x_face_bio_train_'+str(i)+'.npy',x_train)
            np.save('/hdd/sda/lzq/biovid/dataset/dataset/y_face_bio_train_'+str(i)+'.npy',y_train)
        x_train[np.isnan(x_train)] = 0
        y_train[np.isnan(y_train)] = 0
        x_test[np.isnan(x_test)] = 0
        y_test[np.isnan(y_test)] = 0
        regs=['inv','rf']
        for j,reg in enumerate(regs):
            y_pred,y_pred_prob = reg_method(reg,x_train,y_train,x_test,i)
            # Calculate evaluation metrics
            # accuracy = sum(y_pred == y_test) / len(y_test)
            accuracy = accuracy_score(y_test, y_pred)
            # precision = precision_score(y_test, y_pred)
            # recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_prob)
            sub_matric=[accuracy,f1,auc]
            for k,matric in enumerate(matrics):
                matrics[k][j]+=sub_matric[k]

            # Print the evaluation metrics
            print(f"Leave Subject: {i} | Regression Method: {reg}")
            print(f"Accuracy: {accuracy}")
            # print(f"Precision: {precision}")
            # print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"AUC: {auc}")
            print()
    matrics=np.array(matrics)/(end-start)
    print(matrics.T)

bio_pre(74,81)



