import cv2
import numpy as np
import face_alignment
import math
import os
from os.path import basename
from pathlib import Path
import subprocess
import shutil
from pytube import YouTube
from io import BytesIO
from glob import glob
DEVNULL = open(os.devnull, 'wb')
# parpamerters:dict
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=False, face_detector="blazeface")#, device="cpu")


class DownloadException(Exception):
    pass


class InvalidException(Exception):
    pass


def download(uri, parameters):
    print(uri)
    id = uri.rsplit("watch?v=")[-1]
    yt = YouTube(uri)
    video_itag = None
    audio_itag = None
    save_path = f"tmp/{id}"
    videos = sorted(filter(lambda s: s.type == 'video', yt.fmt_streams),
                    key=lambda row: int(row.resolution.replace('p', '')), reverse=True)
    audios = sorted(filter(lambda s: s.type == 'audio', yt.fmt_streams),
                    key=lambda row: int(row.abr.replace('kbps', '')), reverse=True)
    audio_itag = audios[0].itag
    for item in videos:
        if item.resolution == '1080p' and item.fps == 30:
            video_itag = item.itag
            break
    if video_itag is None:
        raise InvalidException("1080p resolution not found or 30fps not found")
    try:
        cmd=["youtube-dl",uri,"-f",str(video_itag)+"+"+str(audio_itag),"-o",save_path]
        subprocess.run(args=cmd, stdout=DEVNULL, stderr=DEVNULL)
    except Exception as e:
        DownloadException("Unable to download")
    save_path = glob(save_path+".*")[-1]
    return save_path, parameters


def video_split(data: str, paramters: dict):
    start,end= paramters['start'],paramters['end']
    x0,y0,w,h= paramters['x0'],paramters['y0'],paramters['w'],paramters['h']
    id = paramters['id']
    # filename = basename(data).rsplit('.')[0]+"_"+str(id)
    filename = Path(data).stem+"_"+str(id)
    filename = os.path.join(f"tmp/video", filename)
    os.makedirs(filename, exist_ok=True)
    video_cmd=['ffmpeg','-i',data,'-ss',str(start),'-to',str(end),'-filter:v', f'crop={w}:{h}:{x0}:{y0}',filename+'/video.mp4']
    audio_cmd=['ffmpeg','-i',data,'-ss',str(start),'-to',str(end),filename+'/audio.wav']
    subprocess.run(video_cmd, stdout=DEVNULL, stderr=DEVNULL)
    subprocess.run(audio_cmd, stdout=DEVNULL, stderr=DEVNULL)
    return filename,paramters


def get_landmark_bbox(data: np.ndarray, parameters: dict):
    max_h, max_w = data.shape[:-1]
    landmarks, bboxes = fa.get_landmarks(data, return_bboxes=True)
    if landmarks is not None:
        index = np.argmax(np.array(bboxes)[:, -1])
        parameters["landmark"] = landmarks[index]
        bbox = bboxes[index][:-1].astype(np.int16)
        bbox_w, bbox_h = bbox[2:]-bbox[:2]
        bbox[0] = math.floor(max(0, bbox[0]-bbox_w*0.15))
        bbox[1] = math.floor(max(0, bbox[1]-bbox_h*0.15))
        bbox[2] = math.ceil(min(max_w-1, bbox[2]+bbox_w*0.15))
        bbox[3] = math.ceil(min(max_h-1, bbox[3]+bbox_h*0.15))
        parameters["bbox"] = bbox
    else:
        raise InvalidException("get_landmark_bbox")
    return data, parameters


def eye_dist(data, parameters):
    eye_dist = 0
    landmark = parameters["landmark"]
    left_eye = (landmark[39]+landmark[36])//2
    right_eye = (landmark[45]+landmark[42])//2
    x_dist, y_dist = right_eye-left_eye
    eye_dist = (y_dist**2+x_dist**2)**0.5
    if eye_dist < 80:
        print(eye_dist)
        raise InvalidException("eye_dist")
    return data, parameters


def get_angle(data, parameters):
    if parameters["valid"]:
        landmark = parameters["landmark"]
        left_eye = (landmark[39]+landmark[36])//2
        right_eye = (landmark[45]+landmark[42])//2
        x_dist, y_dist = right_eye-left_eye
        rotate_angle = math.atan2(y_dist, x_dist) * \
            180/math.pi  # 弧度轉角度 可用math.degree代替
        parameters["angle"] = rotate_angle
    return data, parameters


def rotate_image(data, parameters):
    if parameters["valid"]:
        rotate_center = parameters["landmark"][29]
        M = cv2.getRotationMatrix2D(rotate_center, parameters["angle"], 1)
        h, w = data.shape[:-1]
        data = cv2.warpAffine(data, M, (w, h))
    return data, parameters


def crop_data(data, parameters):
    bbox = parameters["bbox"]  # .astype(np.int16)
    radians = math.radians(parameters['angle'])
    sin_r, cos_r = math.sin(radians), math.cos(radians)
    x_l, y_t, x_r, y_b = bbox
    rotate_center = parameters["landmark"][29]
    x_l = x_l-rotate_center[0]
    y_t = y_t-rotate_center[1]
    x_r = x_r-rotate_center[0]
    y_b = y_b-rotate_center[1]

    new_x1 = cos_r*x_l+sin_r*y_t+rotate_center[0]
    new_x2 = cos_r*x_r+sin_r*y_t+rotate_center[0]
    new_x3 = cos_r*x_l+sin_r*y_b+rotate_center[0]
    new_x4 = cos_r*x_r+sin_r*y_b+rotate_center[0]

    new_y1 = -sin_r*x_l+cos_r*y_t+rotate_center[1]
    new_y2 = -sin_r*x_r+cos_r*y_t+rotate_center[1]
    new_y3 = -sin_r*x_l+cos_r*y_b+rotate_center[1]
    new_y4 = -sin_r*x_r+cos_r*y_b+rotate_center[1]

    min_x = int(max(0, min(new_x1, new_x2, new_x3, new_x4)))
    max_x = int(max(new_x1, new_x2, new_x3, new_x4))
    min_y = int(max(0, min(new_y1, new_y2, new_y3, new_y4)))
    max_y = int(max(new_y1, new_y2, new_y3, new_y4))
    # print(data.shape, min_x, min_y, max_x, max_y)
    data = data[min_y:max_y, min_x:max_x]
    # print(data)
    # data = data[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    return data, parameters

def move_data(data, parameters):
    shutil.move(data, parameters)


def main_pipeline(job_list):
    last_uri = ''
    if not os.path.exists('tmp/video'):
        os.makedirs('tmp/video')
    for i, item in enumerate(job_list):
        try:
            uri = item.pop('uri')
            parameters = item
            if not parameters["valid"]:
                continue
            if 'save_path' in parameters:
                yield parameters['save_path']
                continue
            if last_uri != '' and last_uri != uri:
                if not "https://" in raw_path:
                    print('remove',raw_path)
                    os.remove(raw_path)
            last_uri = uri
            raw_path = uri
            if "https://" in uri:
                tmp = glob('tmp/'+uri.split("watch?v=")[-1]+".*")
                if len(tmp):
                    raw_path = tmp[-1]
                else:
                    print("download")
                    raw_path, parameters = download(uri, parameters)
            video_dir,parameters = video_split(
                raw_path, parameters)
            landmark_list = []
            angle_list = []
            bbox_list = []
            video_path = os.path.join(video_dir, 'video.mp4')
            frame_num = 1
            video = cv2.VideoCapture(video_path)
            try:
                while video.isOpened():
                    ret, frame = video.read()
                    if not ret:
                        break
                    data, parameters = get_landmark_bbox(frame, parameters)
                    data, parameters = eye_dist(data, parameters)
                    data, parameters = get_angle(data, parameters)
                    # data, parameters = rotate_image(data, parameters)
                    # data, parameters = crop_data(data, parameters)
                    landmark_list.append(parameters["landmark"])
                    bbox_list.append(parameters["bbox"])
                    angle_list.append(parameters["angle"])
                    # save_path = os.path.join(
                    #     video_dir, str(frame_num).zfill(5)+".png")
                    # cv2.imwrite(save_path, data)
                    frame_num += 1
                video.release()
                # os.remove(video_path)
                landmark_buffer=BytesIO()
                bbox_buffer=BytesIO()
                angle_buffer=BytesIO()
                np.save(landmark_buffer,landmark_list)
                np.save(bbox_buffer,bbox_list)
                np.save(angle_buffer,angle_list)
                landmark_buffer.seek(0)
                bbox_buffer.seek(0)
                angle_buffer.seek(0)
                np.savez(f"{video_dir}/data", landmark=landmark_list,
                         bbox=bbox_list, angle=angle_list)
                output_dir = f"correct/{basename(video_dir)}"
                move_data(
                    video_dir, output_dir) #move to share dir
                #upload to db landmark,bbox,angle,isdownload=True and save_path
                yield output_dir
            except Exception as e:
                print(e)
                print(video_dir, "is not valid")
                shutil.rmtree(video_dir)
                #valid=false upload to database
        except Exception as e:
            print(e)
            #valid=false upload to database
            # print("download error or split video error")


if __name__ == "__main__":
    job_list = [
        {
            "uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "id": 1, "valid": True,
            "start": 28.84, "end": 35.16,
            'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080,
            # "save_path": "currect/E0NdymcK7wg_1"
        },
        {
            "uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "id": 2, "valid": True,
            "start": 39, "end": 43.44,
            'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080
        },
        {
            "uri": "https://youtube.com/watch?v=sPJ365h2rxI",
            "id": 1, "valid": True,
            "start": 179.64, "end": 205.08,
            'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080
        },
        {
            "uri": "https://youtube.com/watch?v=0Z1r_ATrX9I",
            "id": 1, "valid": True,
            "start": 179.64, "end": 205.08,
            'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080
        },
        {
            "uri": "https://youtube.com/watch?v=02uFohxbJBU",
            "id": 1, "valid": True,
            "start": 13, "end": 20,
            'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080
        },
        {
            "uri": "https://youtube.com/watch?v=02uFohxbJBU",
            "id": 2, "valid": True,
            "start": 15, "end": 32,
            'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080
        },
    ]
    for item in main_pipeline(job_list):
        print("item", item)
        #read_video
        