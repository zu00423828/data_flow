import cv2
import  numpy as np
import face_alignment
import math
import os
import subprocess


#parpamerters:dict


fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=False, face_detector="blazeface", device="cpu")

class DownloadException(Exception):
    def __init__(self):
        super().__init__("unable to download")
class ValidException(Exception):
    def __init__(self) -> None:
        super().__init__("is not valid")
def download(uri):
    id=uri.rsplit("https://youtube.com/watch?v=")[-1]
    try:
        subprocess.run(f"youtube-dl '{uri}'  -f 136+140 -o 'tmp/video/{id}.mp4'",shell=False,check=True)
    except Exception :
        print("發生錯誤")
def avspeech_preprocess(data:str,args:dict):
    id,start,end=[]
    path=os.path.join(f"tmp/video",data)
    os.makedirs(data,exist_ok=True)
    subprocess.run(f"ffmpeg -i '{data} --ss {start} -to {end} {path}/%5d.png'")
    subprocess.run(f"ffmpeg -i '{data} --ss {start} -to {end} {path}/audio.wav'")
def voxceleb2_preproces(data: str,args:dict):
    id,start,end,x,y=[]
    path=os.path.join(f"tmp/video",data)
    subprocess.run(f"ffmpeg -i '{data} --ss {start} -to {end} {path}/%5d.png'")
    subprocess.run(f"ffmpeg -i '{data} --ss {start} -to {end} {path}/audio.wav'")
def get_landmark_bbox(data:np.ndarray,parameters:dict):
    max_h,max_w=data.shape[:-1]
    landmarks,bboxes=fa.get_landmarks(data,return_bboxes=True)
    if landmarks is not None:
        parameters["landmark"]=landmarks[-1]
        bbox=bboxes[-1][:-1].astype(np.int16)
        bbox_w,bbox_h=bbox[2:]-bbox[:2]

        bbox[0]=max(0,bbox[0]-bbox_w*0.1)
        bbox[1]=max(0,bbox[1]-bbox_h*0.1)
        bbox[2]=min(max_w-1,bbox[2]+bbox_w*0.1)
        bbox[3]=min(max_h-1,bbox[3]+bbox_h*0.1)

        parameters["bbox"]=bbox
    else:
        parameters["vaild"]=False
        raise ValidException()
    return data,parameters
def eye_dist(data,parameters):
    eye_dist=0
    if parameters["valid"]:
        landmark=parameters["landmark"]
        left_eye=(landmark[39]+landmark[36])//2
        right_eye=(landmark[45]+landmark[42])//2
        x_dist,y_dist=right_eye-left_eye
        eye_dist=(y_dist**2+x_dist**2)**0.5
    if eye_dist < 80:
        parameters["valid"]=False
        raise ValidException()
    return data,parameters
def crop_data(data,parameters):
    if parameters["valid"]:
        bbox=parameters["bbox"].astype(np.int16)
        data=data[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        return data,parameters
def get_angle(data,parameters):
    if parameters["valid"]:
        landmark=parameters["landmark"]
        bbox=parameters["bbox"]
        left_eye=(landmark[39]+landmark[36])//2
        right_eye=(landmark[45]+landmark[42])//2
        x_dist,y_dist=right_eye-left_eye
        rotate_angle=math.atan2(y_dist,x_dist)*180/math.pi
        parameters["angle"]=rotate_angle
    return data,parameters
def rotate_image(data,parameters):
    if parameters["valid"]:
        bbox=parameters["bbox"]
        offset=bbox[:2]
        desiredLeftEye= ( 0.4 , 0.4 )
        rotate_center=parameters["landmark"][29]#-offset
        # rotate_center=(parameters["landmark"][39]+parameters["landmark"][42])/2#-offset
        cv2.circle(data,tuple(rotate_center.astype(np.int16)),1,(255,255,255),1)
        M=cv2.getRotationMatrix2D(rotate_center,parameters["angle"],1)
        h,w=data.shape[:-1]
        fw,fh=bbox[2:]-bbox[:2]
        tx=fw*0.5
        ty=fh*0.4
        print(tx,ty)
        M[0,2]+=(tx-rotate_center[0])
        M[1,2]+=(ty-rotate_center[1])
        data=cv2.warpAffine(data,M,(int(fw),int(fh)))
    return data,parameters
if __name__=="__main__":
    parameters={"valid":True,"is_down":True}
    video_list=["test.mp4"]
    for item in video_list:
        landmark_list=[]
        angle_list=[]
        video=cv2.VideoCapture(item)
        dirname=item.rsplit(".mp4")[0]
        os.makedirs(dirname,exist_ok=True)
        try:
            count=1
            while video.isOpened():
                ret,frame=video.read()
                if not ret:
                    break
                print(count)
                data,parameters=get_landmark_bbox(frame,parameters)
                
                data,parameters=eye_dist(data,parameters)
                data,parameters=crop_data(data,parameters)
                data,parameters=get_angle(data,parameters)
                landmark_list.append(parameters["landmark"])
                angle_list.append(parameters["angle"])
                save_path=dirname+"/"+str(count).zfill(5)+".png"
                cv2.imwrite(save_path,data)
                count+=1
        except:
            print("error")
        if parameters["valid"]:
            np.savez(f"{item}.npy",landmark=landmark_list,angle=angle_list)
        
