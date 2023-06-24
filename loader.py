from posixpath import split
import time
import cv2
import torch
import numpy as np
from PIL import Image
from utils import *
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision.transforms import ToTensor, Resize, RandomCrop,Compose,RandomHorizontalFlip,RandomVerticalFlip,Normalize,ColorJitter,ToPILImage
import random
import os
from models import Prototype,Classifier,ResNet18,cnn1d,VGG,Regressor
import torchvision.transforms.functional as tf
from math import *
from torchvision.transforms.functional import rotate
from scipy import signal
from torch.nn.functional import conv1d

class AllDataset(Dataset):
    def __init__(self,is_train,train_rio,path,modal,is_time,pic_size,is_all_modal=False,leave_subject=80):
        self.is_time=is_time
        self.modal=0 if modal=='face' else 1 if modal=='gsr' else 2 if modal=='ecg' else 3 if modal=='emg' else 6 if modal=='bio' else -1
        self.is_all_modal=is_all_modal
        self.all_items=[]
        self.items=[]
        self.pic_transform = Compose([
                Resize([pic_size,pic_size]),
                ToTensor()])
        
        self.all_items=getAllSample(path,0,4)
        # train_int=int(len(self.all_items)*train_rio)
        tmp=list(self.all_items.values())
        if leave_subject==-1:
            for i in range(len(tmp)):
                train_int=int(len(tmp[i])//2*train_rio)
                start_int=len(tmp[i])//2
                tmp[i]=tmp[i][:train_int]+tmp[i][start_int:start_int+train_int] if is_train else tmp[i][train_int:start_int]+tmp[i][start_int+train_int:]
                self.items.append(tmp[i])
        else:
            self.items=tmp[:leave_subject]+tmp[leave_subject+1:] if is_train else tmp[leave_subject:leave_subject+1]
            print(list(self.all_items.keys())[leave_subject])
        self.all_items=self.items
        self.items=[]
        
        for items in self.all_items:
            self.items+=items
        self.all_items=self.items
        self.items=[]

        if not is_time:
            houzhui='jpg'
            for items in self.all_items:
                # if len(os.listdir(items[0]))!=28*2:
                #     print(items[0])
                item_list=os.listdir(items[0])
                item_list.remove('imgs.pt')
                for img in sorted(item_list,key=lambda x:int(x.split('.')[0])):
                    if img.endswith(houzhui):
                        self.items.append([os.path.join(items[0],img),items[-1]])
            self.all_items=self.items
        print(len(self.all_items))

    def __len__(self):
        return len(self.all_items)

    def load_rgb(self,file_path):
        img = Image.open(file_path)
        img = self.pic_transform(img)
        return img
    
    def get_y_label(self):
        y_label=[0,0,0,0,0]
        for item in self.all_items:
            y_label[int(item[-1])]+=1
        self.y_label=y_label
        print(y_label)

    def normalized(self,x):
        mean=x.mean()
        std=x.std()
        nor=(x-mean)/std
        return torch.nan_to_num(nor)

    def butter_bandpass_filter(self,data, lowcut, highcut, fs):
        data=np.array(data)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(3, [low, high], btype='band') if high>0 else signal.butter(3, low, btype='low')
        y = signal.filtfilt(b, a, data)
        return y

    # def load_gsr_ecg_emg(self,gsr,ecg,emg):
    #     gsr=self.load_gsr(gsr)
    #     ecg=self.load_ecg(ecg)
    #     emg=self.load_emg(emg)
    #     return torch.cat([gsr,ecg,emg],dim=-1)

    def bio_normal(self,bio):
        # bio=self.normalized(bio)
        # window_size=32
        # bio = bio.view(1, 1, -1)
        # weights = torch.ones(1, 1, window_size) / window_size
        # bio = conv1d(bio, weights)
        even_indices = torch.arange(0, bio.size(0), 2)
        bio = bio[even_indices]
        # bio = bio.view(-1)
        return bio.unsqueeze(-1)

    def load_gsr(self,gsr):
        gsr=self.butter_bandpass_filter(gsr, 0.2,-1, 512) 
        gsr=torch.Tensor(gsr.copy())
        gsr=self.bio_normal(gsr)
        return gsr
    
    def load_ecg(self,ecg):
        ecg = self.butter_bandpass_filter(ecg, 0.1, 250, 512) 
        p = np.polyfit(np.arange(len(ecg)), ecg, 5)
        fitted = np.polyval(p, np.arange(len(ecg)))
        ecg = ecg - fitted
        ecg=torch.Tensor(ecg.copy())
        ecg=self.bio_normal(ecg)
        return ecg

    def load_emg(self,emg):
        emg = self.butter_bandpass_filter(emg, 20, 250, 512)
        emg=torch.Tensor(emg.copy())
        emg=self.bio_normal(emg)
        return emg
    
    def normal_vector(self,p1, p2, p3):
        v1 = p3 - p1
        v2 = p2 - p1
        cp = np.cross(v1, v2)[0]
        return cp
    
    def dis(self,p1,p2):
        ans=0
        for i in range(p1.shape[0]):
            ans+=(p1[i]-p2[i])**2
        return sqrt(ans)

    def load_face_point(self,file_path):
        npy=np.load(file_path)
        #eyes
        eyes=[(npy[43]+npy[44])/2.0,(npy[46]+npy[47])/2.0,(npy[37]+npy[38])/2.0,(npy[40]+npy[41])/2.0]
        noses=[np.mean(npy[31:35],axis=0)]
        nv=[self.normal_vector(eyes[3],noses,eyes[1]),eyes[3]-eyes[1]]
        nvs=[x/np.linalg.norm(x) for x in nv]
        aus=[
            #eye
            self.dis(eyes[0],eyes[1]),
            self.dis(eyes[2],eyes[3]),
            #eye-brow
            self.dis(eyes[0],npy[24]),
            self.dis(eyes[2],npy[19]),
            #eye-mouth
            self.dis(eyes[1],npy[64]),
            self.dis(eyes[3],npy[60]),
            #blow-mouth
            self.dis(npy[24],npy[64]),
            self.dis(npy[19],npy[60]),
            #mouth
            self.dis(npy[60],npy[64]),
            self.dis(npy[51],npy[57]),
        ]
        # gra=self.gra_helper(file_path)
        return [np.array(x) for x in [nvs,aus]]

    def __getitem__(self,idr):
        item=self.all_items[idr]
        imgs_return=[[],[],[],[],[],[],[],[],item[0]]
        label=int(item[-1])
        bio_funs=[self.load_gsr,self.load_ecg,self.load_emg,self.load_emg,self.load_emg]
        if self.is_time:
            if self.modal==0:
                # if not os.path.exists(os.path.join(item[0],'imgs.pt')):
                listdir=os.listdir(item[0])
                listdir.remove('imgs.pt')
                houzhui='jpg'
                for img in sorted(listdir,key=lambda x:int(x.split('.')[0])):
                    if img.endswith(houzhui):
                        img=self.load_rgb(os.path.join(item[0],img))
                        imgs_return[0].append(img)
                while(len(imgs_return[0])<28):
                    copy=imgs_return[0][-1].clone()
                    imgs_return[0].append(copy)

                imgs_return[0]=torch.stack(imgs_return[0])

                ###########################################
                
                
                #     torch.save(imgs_return[0], os.path.join(item[0],'imgs.pt'))
                    
                # else:
                #     imgs_return[0]=torch.load(os.path.join(item[0],'imgs.pt'))
            elif self.modal==6:
                for i in range(1,6):
                    imgs_return[i]=bio_funs[i-1](item[i])
                imgs_return[6]=torch.cat(imgs_return[1:6],dim=-1)

            elif self.modal==-1:
                imgs_return[0]=torch.load(os.path.join(item[0],'imgs.pt'))
                for i in range(1,6):
                    imgs_return[i]=bio_funs[i-1](item[i])
                imgs_return[6]=torch.cat(imgs_return[1:6],dim=-1)
                listdir=os.listdir(item[0])
                listdir.remove('imgs.pt')
                houzhui='npy'
                for img in sorted(listdir,key=lambda x:int(x.split('.')[0])):
                    if img.endswith(houzhui):
                        imgs_return[7].append(self.load_face_point(os.path.join(item[0],img)))

            else:
                imgs_return[self.modal]=bio_funs[self.modal-1](item[self.modal])

        else:
            # if self.modal==0:
            img=self.load_rgb(item[0])
            imgs_return[0].append(img)

        return {'xs': imgs_return,'y':label}


def main():
    DATA_PATH='/hdd/sda/lzq/biovid/dataset/label.csv'
    train_dataset=AllDataset(is_train=1,train_rio=0.8,path=DATA_PATH,modal='face',is_time=0,pic_size=128)
    train_dataset.get_y_label()
    train_dataloader = DataLoader(train_dataset, batch_size=4,shuffle=True)
    for data in train_dataloader:
        pass

