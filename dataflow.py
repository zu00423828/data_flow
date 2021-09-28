from youtube_speech import YoutubeSpeechDB
import sys
import cv2
import numpy as np
import face_alignment
import math
import os
import hashlib
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
    face_alignment.LandmarksType._2D, flip_input=False, face_detector="blazeface")  # , device="cpu")


class DownloadException(Exception):
    pass


class InvalidException(Exception):
    pass

class MovefileException(Exception):
    pass

def download(uri, parameters, download_path):
    print(uri)
    id = uri.rsplit("watch?v=")[-1]
    yt = YouTube(uri)
    video_itag = None
    audio_itag = None
    save_path = f"{download_path}/{id}"
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
        cmd = ["youtube-dl", uri, "-f",
               str(video_itag)+"+"+str(audio_itag), "-o", save_path]
        subprocess.run(args=cmd, check=True, stdout=DEVNULL, stderr=DEVNULL)
    except Exception as e:
        DownloadException("Unable to download")
    if len(glob(save_path+".*")):
        save_path = glob(save_path+".*")[-1]
    else:
        raise DownloadException("Unable to download")
    return save_path, parameters


def video_split(data: str, paramters: dict, tmp_path):
    start, end = paramters['start'], paramters['end']
    x0, y0, w, h = paramters['x0'], paramters['y0'], paramters['w'], paramters['h']
    id = paramters['split']
    # filename = basename(data).rsplit('.')[0]+"_"+str(id)
    filename = Path(data).stem+"_"+str(id)+'.mp4'
    filename = os.path.join(tmp_path, filename)
    # os.makedirs(filename, exist_ok=True)
    video_cmd = ['ffmpeg', '-i', data, '-ss', str(start), '-to', str(
        end), '-filter:v', f'crop={w}:{h}:{x0}:{y0}', filename, '-n']  # +'/video.mp4']
    # audio_cmd=['ffmpeg','-i',data,'-ss',str(start),'-to',str(end),filename+'/audio.wav']
    subprocess.run(video_cmd, stdout=DEVNULL, stderr=DEVNULL)
    # subprocess.run(audio_cmd, stdout=DEVNULL, stderr=DEVNULL)
    return filename, paramters


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


def move_data(data, output_path):
    subprocess.run(['cp',data,output_path],stdout=DEVNULL,stderr=DEVNULL)
    source_md5=hashlib.md5(open(data,'rb').read()).hexdigest()
    target_md5=hashlib.md5(open(output_path,'rb').read()).hexdigest()
    if source_md5!=target_md5:
        print(source_md5,target_md5)
        os.remove(output_path)
        raise MovefileException('md5 is different')
    else:
        os.remove(data)

def main_pipeline(share_root, download_path='/tmp/', tmp_path='/tmp/video/'):
    if os.path.isdir(share_root):
        db_path=os.path.join(share_root,'youtube_speech.sqlite')
        correct_path=os.path.join(share_root,'correct')
    else:
        db_path=share_root
        correct_path=os.path.join(os.path.dirname(),'correct')
    last_uri = ''
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    if not os.path.exists(correct_path):
        os.makedirs(correct_path)
    db = YoutubeSpeechDB(db_path)
    while True:
        with db.session(limit=20, dataset_type=None) as sess:
            jobs = db.list_jobs(processing_ticket_id=sess.processing_ticket_id)
            assert len(jobs) > 0
            for job in jobs:
                try:
                    id = job.pop('id')
                    uri = job.pop('uri')
                    parameters = job
                    if not parameters["valid"]:
                        continue
                    if parameters['path'] is not None:
                        yield parameters['path'], parameters['landmarks'], parameters['bboxes'], parameters['angles']
                        continue
                    if last_uri != '' and last_uri != uri:
                        if not "https://" in raw_path:
                            print('remove', raw_path)
                            os.remove(raw_path)
                    last_uri = uri
                    raw_path = uri
                    if "https://" in uri:
                        tmp = glob(download_path +
                                   uri.split("watch?v=")[-1]+".*")
                        if len(tmp):
                            raw_path = tmp[-1]
                        else:
                            print("download")
                            raw_path, parameters = download(
                                uri, parameters, download_path)
                    video_path, parameters = video_split(
                        raw_path, parameters, tmp_path)
                    landmark_list = []
                    angle_list = []
                    bbox_list = []
                    frame_num = 1
                    video = cv2.VideoCapture(video_path)
                    fps=video.get(5)
                    frame_count=video.get(7)
                    print(fps,frame_count)
                    try:
                        while video.isOpened():
                            ret, frame = video.read()
                            if not ret:
                                break
                            data, parameters = get_landmark_bbox(
                                frame, parameters)
                            data, parameters = eye_dist(data, parameters)
                            data, parameters = get_angle(data, parameters)
                            # data, parameters = rotate_image(data, parameters)
                            # data, parameters = crop_data(data, parameters)
                            landmark_list.append(parameters["landmark"])
                            bbox_list.append(parameters["bbox"])
                            angle_list.append(parameters["angle"])
                            frame_num += 1
                        video.release()
                        landmark_buffer = BytesIO()
                        bbox_buffer = BytesIO()
                        angle_buffer = BytesIO()
                        np.save(landmark_buffer, landmark_list)
                        np.save(bbox_buffer, bbox_list)
                        np.save(angle_buffer, angle_list)
                        landmark_buffer.seek(0)
                        bbox_buffer.seek(0)
                        angle_buffer.seek(0)
                        output_path = os.path.join(
                            correct_path, basename(video_path))
                        move_data(
                            video_path, output_path)  # move to share dir
                        # upload to db landmark,bbox,angle,isdownload=True and save_path
                        db.update_job(youtube_speech_id=id, valid=True, path=output_path, landmarks=landmark_buffer.read()
                            , bboxes=bbox_buffer.read(), angles=angle_buffer.read(),fps=fps,frame_count=frame_count)
                        yield video_path, np.array(landmark_list), np.array(bbox_list), np.array(angle_list)
                    except MovefileException:
                        os._exit(0)
                    except Exception as e:
                        video.release()
                        print(e)
                        db.update_job(youtube_speech_id=id, valid=False)
                        print(video_path, "is invalid")
                        os.remove(video_path)
                        # valid=false upload to database
                except Exception as e:
                    print(e)
                    if str(e) in "HTTP Error 429: Too Many Requests":
                        continue
                    else:
                        db.update_job(youtube_speech_id=id, valid=False)
                    # valid=false upload to database
                    print("download error or split video error")


if __name__ == "__main__":
    share_root = '/home/yuan/share/youtube-speech/'
    download_path = '/tmp/'
    tmp_path = '/tmp/video/'
    for item in main_pipeline(share_root, download_path, tmp_path):
        pass