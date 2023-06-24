import cv2
import os
import face_alignment
from skimage import io
import numpy as np
import shutil
from tqdm import tqdm
import csv
import ssl
 
ssl._create_default_https_context = ssl._create_unverified_context

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False,device="cuda")

def is_iou(x1,x2,y1,y2,x1_pre,x2_pre,y1_pre,y2_pre):
    area1=(x2-x1)*(y2-y1)
    area2=(x2_pre-x1_pre)*(y2_pre-y1_pre)
    if min(area1,area2)/max(area1,area2)<0.1:
        return False
    xs=[x1_pre,x2_pre]
    ys=[y1_pre,y2_pre]
    for x in xs:
        for y in ys:
            if x>=x1 and x<=x2 and y>=y1 and y<=y2:
                return True
    xs=[x1,x2]
    ys=[y1,y2]
    for x in xs:
        for y in ys:
            if x>=x1_pre and x<=x2_pre and y>=y1_pre and y<=y2_pre:
                return True
    return False

def face_points_detect(img,x1_pre,x2_pre,y1_pre,y2_pre):
    
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_h=img.shape[0]
    img_w=img.shape[1]
    faces = fa.get_landmarks(img)
    if faces==None:
        return None,None,x1_pre,x2_pre,y1_pre,y2_pre

    x1,x2,y1,y2=0,0,0,0
    maxn=0
    pos=None
    for face in faces:
        xx1=int(np.min(face[:,0]))
        xx2=int(np.max(face[:,0]))
        yy1=int(np.min(face[:,1]))
        yy2=int(np.max(face[:,1]))
        w=xx2-xx1
        h=yy2-yy1
        if h*w>maxn:
            x1,x2,y1,y2=xx1,xx2,yy1,yy2
            maxn=h*w
            pos=face
            if w<h:
                cha=(h-w)//2
                x1=max(x1-cha,0)
                x2=min(x2+cha,img_w)
            else:
                cha=(w-h)//2
                y1=max(y1-cha,0)
                y2=min(y2+cha,img_h)

    if not ((x1_pre,x2_pre,y1_pre,y2_pre)==(0,0,0,0) or is_iou(x1,x2,y1,y2,x1_pre,x2_pre,y1_pre,y2_pre)):
        return None,None,x1_pre,x2_pre,y1_pre,y2_pre
    
    face_img=img[y1:y2,x1:x2]

    return face_img,pos,x1,x2,y1,y2

def mp42img(path):
    dir_path=os.path.join(path,'video')
    save_path=os.path.join(path,'face')
    bio_path=os.path.join(path,'bio')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(os.path.join(path,'label2.csv'),'w',encoding='utf-8') as f:
        csv_writer = csv.writer(f)

        bar_out = tqdm(total=len(os.listdir(dir_path)), desc='person')
        bar = tqdm(total=100, desc='video')
        for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('_')[0])):
            bar_out.set_postfix(**{'id':i})
            bar_out.update(1)

            dir_path_2=os.path.join(dir_path,i)
            bio_path_2=os.path.join(bio_path,i)

            for j in sorted(os.listdir(dir_path_2)):
                bar.set_postfix(**{'id':j})
                bar.update(1)

                dir_path_3=os.path.join(dir_path_2,j)
                bio_path_3=os.path.join(bio_path_2,j[:-4]+'_bio.csv')
                save_path_2=os.path.join(save_path,i)
                if not os.path.exists(save_path_2):
                    os.mkdir(save_path_2)
                save_path_2=os.path.join(save_path_2,j)

                subject_id=i
                face_row=save_path_2
                bio_row=bio_path_3 if os.path.exists(bio_path_3) else ""
                label=j[:-4].split('-')[-2]
                label=0 if label[:2]=='BL' else int(label[-1])
                row=[subject_id,face_row,bio_row,label]
                csv_writer.writerow(row)

                if not os.path.exists(save_path_2):
                    os.mkdir(save_path_2)
                else:
                    continue
                    
                cap = cv2.VideoCapture(dir_path_3)
                fps = cap.get(cv2.CAP_PROP_FPS)

                x1,x2,y1,y2=0,0,0,0
                cnt=-1
                actcnt=-1
                while cap.isOpened():
                    actcnt+=1
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if actcnt%round(fps//5)!=0:
                        continue
                    cnt+=1
                    img,pos,x1,x2,y1,y2=face_points_detect(frame,x1,x2,y1,y2)
                    if img is None or img.shape[0]*img.shape[1]*img.shape[2]==0:
                        continue
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_path_2,str(cnt))+".jpg",img)
                    np.save(os.path.join(save_path_2,str(cnt)+".npy"),pos)
                cap.release()

            bar.update(-100)
        bar_out.close()

     
mp42img('/hdd/sda/lzq/biovid/dataset')


def get_video_name(paths):
    #f = open('/hdd/sdd/lzq/DLMM_new/dataset/video_name.csv', 'w')
    #writer = csv.writer(f)
    for path in paths:
        dir_path=os.path.join(path,'video')
        bar = tqdm(total=len(os.listdir(dir_path)), desc=f"face path: {dir_path}")
        for i in sorted(os.listdir(dir_path),key=lambda x:int(x)):
            dir_path_2=os.path.join(dir_path,i)
            #videos=["","","",""]
            for j in sorted(os.listdir(dir_path_2)):
                # if int(j.split('.')[0][-1])<5:
                #     videos[int(j.split('.')[0][-1])-1]='/mp4/'+j
                dir_path_3=os.path.join(dir_path_2,j)
                os.system("cp "+dir_path_3+" /hdd/sda/lzq/mp4")
            #writer.writerow(videos)
            bar.set_postfix(**{'person': i})
            bar.update(1)
        bar.close()
    #f.close()

def get_bio_name(paths):
    # bio_list=[]
    # for i in range(1050):
    #     bio_list.append(["","","",""])
    # f = open('/hdd/sdd/lzq/DLMM_new/dataset/bio_name.csv', 'w')
    # writer = csv.writer(f)
    for path in paths:
        dir_path=os.path.join(path,'bio')
        bar = tqdm(total=len(os.listdir(dir_path)), desc=f"face path: {dir_path}")
        for i in sorted(os.listdir(dir_path),key=lambda x:int(x.split('-')[0])):
            if int(i.split('.')[0][-1])<5:
                #bio_list[int(i.split('-')[0])][int(i.split('.')[0][-1])-1]='/csv/'+i
                dir_path_3=os.path.join(dir_path,i)
                os.system("cp "+dir_path_3+" /hdd/sda/lzq/csv")
            bar.set_postfix(**{'person': i})
            bar.update(1)
        bar.close()
    # writer.writerows(bio_list)
    # f.close()

#get_bio_name([
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain1",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain2",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain3",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.5/pain4",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain3",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain4",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain7",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain8",
#     "/hdd/sdd/lzq/DLMM_new/dataset/2022.3.23/pain5",
#   ])