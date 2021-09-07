from sys import path
import cv2
import numpy as np
import face_alignment
import math
import os
import subprocess
import shutil
DEVNULL = open(os.devnull, 'wb')
# parpamerters:dict
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=False, face_detector="blazeface", device="cuda")


class DownloadException(Exception):
    def __init__(self):
        super().__init__("unable to download")


class ValidException(Exception):
    def __init__(self) -> None:
        super().__init__("is not valid")


def download(uri, parameters):
    print(uri)
    id = uri.rsplit("https://youtube.com/watch?v=")[-1]
    try:
        save_path = f"tmp/video/{id}.mp4"
        subprocess.run(
            f"youtube-dl '{uri}'  -f 136+140 -o 'tmp/video/{id}.mp4'", shell=False, check=True)
    except Exception:
        print("沒有1080p，或無法下載")
        raise DownloadException()
    return save_path, parameters


def avspeech_preprocess(data: str, args: dict):
    id, start, end = [0, 3, 10]
    path = os.path.join(f"tmp/process", data)
    os.makedirs(data, exist_ok=True)
    subprocess.run(
        f"ffmpeg -i '{data}' -ss {start} -to {end} '{path}/%5d.png'")
    subprocess.run(
        f"ffmpeg -i '{data}' -ss {start} -to {end} '{path}/audio.wav'")


def voxceleb2_preproces(data: str, paramters: dict):
    timestamp_list=paramters["timestamp"]
    video_path=[]
    for i,item in enumerate(timestamp_list): 
        filename = data.rsplit(".mp4")[0]+"_"+str(i)
        filename = os.path.join(f"tmp/video", filename)
        os.makedirs(filename, exist_ok=True)
        subprocess.call(f"ffmpeg -i {data} -ss {item['start']} -to {item['end']} {filename}/%5d.png",
                        shell=True, stdout=DEVNULL, stderr=DEVNULL)
        # print(f"ffmpeg -i {data} -ss {item['start']} -to {item['end']} {filename}/%5d.png")
        subprocess.call(f"ffmpeg -i {data} -ss {item['start']} -to {item['end']} {filename}/audio.wav",
                        shell=True, stdout=DEVNULL, stderr=DEVNULL)
        video_path.append(filename)
    return video_path


def get_landmark_bbox(data: np.ndarray, parameters: dict):
    max_h, max_w = data.shape[:-1]
    landmarks, bboxes = fa.get_landmarks(data, return_bboxes=True)
    if landmarks is not None:
        parameters["landmark"] = landmarks[-1]
        bbox = bboxes[-1][:-1].astype(np.int16)
        bbox_w, bbox_h = bbox[2:]-bbox[:2]

        bbox[0] = max(0, bbox[0]-bbox_w*0.1)
        bbox[1] = max(0, bbox[1]-bbox_h*0.1)
        bbox[2] = min(max_w-1, bbox[2]+bbox_w*0.1)
        bbox[3] = min(max_h-1, bbox[3]+bbox_h*0.1)

        parameters["bbox"] = bbox
    else:
        parameters["vaild"] = False
        raise ValidException()
    return data, parameters


def eye_dist(data, parameters):
    eye_dist = 0
    landmark = parameters["landmark"]
    left_eye = (landmark[39]+landmark[36])//2
    right_eye = (landmark[45]+landmark[42])//2
    x_dist, y_dist = right_eye-left_eye
    eye_dist = (y_dist**2+x_dist**2)**0.5
    if eye_dist < 80:
        parameters["valid"] = False
        raise ValidException()
    return data, parameters


def crop_data(data, parameters):
    bbox = parameters["bbox"].astype(np.int16)
    data = data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return data, parameters


def get_angle(data, parameters):
    if parameters["valid"]:
        landmark = parameters["landmark"]
        bbox = parameters["bbox"]
        left_eye = (landmark[39]+landmark[36])//2
        right_eye = (landmark[45]+landmark[42])//2
        x_dist, y_dist = right_eye-left_eye
        rotate_angle = math.atan2(y_dist, x_dist)*180/math.pi
        parameters["angle"] = rotate_angle
    return data, parameters


def rotate_image(data, parameters):
    if parameters["valid"]:
        bbox = parameters["bbox"]
        offset = bbox[:2]
        desiredLeftEye = (0.4, 0.4)
        rotate_center = parameters["landmark"][29]  # -offset
        # rotate_center=(parameters["landmark"][39]+parameters["landmark"][42])/2#-offset
        cv2.circle(data, tuple(rotate_center.astype(np.int16)),
                   1, (255, 255, 255), 1)
        M = cv2.getRotationMatrix2D(rotate_center, parameters["angle"], 1)
        h, w = data.shape[:-1]
        fw, fh = bbox[2:]-bbox[:2]
        tx = fw*0.5
        ty = fh*0.4
        print(tx, ty)
        M[0, 2] += (tx-rotate_center[0])
        M[1, 2] += (ty-rotate_center[1])
        data = cv2.warpAffine(data, M, (int(fw), int(fh)))
    return data, parameters


def move_data(data, parameters):
    shutil.move(data, parameters)


if __name__ == "__main__":
    # job_list=["test.mp4","https://www.youtube.com/watch?v=xWTiOqJqkk0"]
    # parameters_list=[{"valid":True,"is_download":False,"is_avspeech":False},
    #     {"valid":True,"is_download":True,"is_avspeech":True}]
    job_list = [{"uri": "test.mp4", 
    "parameters": {"valid": True,"is_avspeech":False,
    "timestamp": [{"start": 1, "end": 5}, {"start": 6, "end": 10}, {"start": 11, "end": 15}]
    }}
    ,{"uri": "https://www.youtube.com/watch?v=xWTiOqJqkk0", 
    "parameters": {"valid": True,"is_avspeech":True,
    "timestamp": [{"start": 1, "end": 5}, {"start": 6, "end": 10}, {"start": 11, "end": 15}]
    }}
    ]
    for i, item in enumerate(job_list):
        # for key,value in item.items():
        #     print(key,value)
        # try:
            raw_path=item["uri"]
            paramters=item["parameters"]
            if "https://" in item["uri"]:
                raw_path = download(item["uri"],paramters )
            if paramters["is_avspeech"]:
                video_dir_list = avspeech_preprocess(
                    raw_path,paramters )
            else:
                video_dir_list = voxceleb2_preproces(
                    raw_path,paramters)
            for video_dir in video_dir_list:
                landmark_list = []
                angle_list = []
                bbox_list = []
                print(video_dir)
                for frame_path in os.listdir(video_dir):
                    parameters = {"valid": True}
                    if not frame_path.endswith(".png"):
                        continue
                    frame = cv2.imread(os.path.join(video_dir, frame_path))
                    data, parameters = get_landmark_bbox(frame, parameters)
                    data, parameters = eye_dist(data, parameters)
                    data, parameters = crop_data(data, parameters)
                    data, parameters = get_angle(data, parameters)
                    landmark_list.append(parameters["landmark"])
                    bbox_list.append(parameters["bbox"])
                    angle_list.append(parameters["angle"])
                    save_path = os.path.join(video_dir, frame_path)
                    # print('save_path',save_path)
                    cv2.imwrite(save_path, data)
                if parameters["valid"]:
                    np.savez(f"{video_dir}/data", landmark=landmark_list,
                             bbox=bbox_list, angle=angle_list)
                    move_data(video_dir, f"currect/{os.path.basename(video_dir)}")
        # except Exception as e:
        #     print(e)
            # print("is not valid")
