import numpy as np
from scipy.fftpack import fft
import os
import yaml
import csv
import math
import scipy.linalg as linalg
from time import time
import pandas as pd

def getAllSample(label_path,vs1,vs2):
    samples={}
    cnt_arr=[0,0,0,0,0]
    with open(label_path,'r',encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        for line in csv_reader:
            if int(line[3]) not in [vs1,vs2]:
                continue
            label=0 if int(line[3])==0 else 1
            bios=[[] for _ in range(5)]
            t1=time()
            df = pd.read_csv(line[2], sep='\t', header=None,skiprows=1)
            bios = [df[i].tolist() for i in range(1, 6)]
            # with open(line[2], 'r') as f:
            #     reader = csv.reader(f)
            #     next(reader)
            #     for row in reader:
            #         for column_index in range(1,6):
            #             bios[column_index-1].append(float(row[0].split('\t')[column_index]))
            t2=time()
            if not line[0] in samples.keys() :
                samples[line[0]]=[[line[1],*bios,label]]
            else:
                samples[line[0]].append([line[1],*bios,label])
            cnt_arr[label]+=1
    # print(cnt_arr)
    return samples

def getCfg(yaml_path):
    with open(yaml_path,"r",encoding="utf-8") as f:
        cfg=f.read()
        cfg=yaml.load(cfg,Loader=yaml.FullLoader)
        return cfg
    
# getAllSample('/hdd/sda/lzq/biovid/dataset/label.csv')