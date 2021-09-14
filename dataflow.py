import cv2
import numpy as np
import face_alignment
import math
import os
from os.path import basename
import subprocess
import shutil
DEVNULL = open(os.devnull, 'wb')
# parpamerters:dict
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=False, face_detector="sfd", device="cuda")


class DownloadException(Exception):
    def __init__(self):
        super().__init__("unable to download")


class ValidException(Exception):
    def __init__(self) -> None:
        super().__init__("is not valid")


def download(uri, parameters):
    print(uri)
    id = uri.rsplit("watch?v=")[-1]
    try:
        save_path = f"tmp/video/{id}.mp4"
        subprocess.run(
            f"youtube-dl '{uri}'  -f 137+140 -o 'tmp/video/{id}.mp4'", shell=True, stdout=DEVNULL, stderr=DEVNULL)
    except Exception:
        raise DownloadException()
    return save_path, parameters


def avspeech_preprocess(data: str, paramters: dict):
    timestamp= paramters["timestamp"]
    crop= paramters["bbox_crop"]
    id=paramters['id']
    filename = basename(data).rsplit('.mp4')[0]+"_"+str(id)
    filename = os.path.join(f"tmp/video", filename)
    os.makedirs(filename, exist_ok=True)
    subprocess.call(f"ffmpeg -i {data} -ss {timestamp['start']} -to {timestamp['end']} -filter:v 'crop={crop['w']}:{crop['h']}:{crop['x']}:{crop['y']}' \
        {filename}/video.mp4", shell=True, stdout=DEVNULL, stderr=DEVNULL)
    subprocess.call(f"ffmpeg -i {data} -ss {timestamp['start']} -to {timestamp['end']} {filename}/audio.wav",
                    shell=True, stdout=DEVNULL, stderr=DEVNULL)
    return filename


def voxceleb2_preproces(data: str, paramters: dict):
    from itertools import islice
    video_path = []
    split_txt_dir = 'txt/E0NdymcK7wg'
    for i, item in enumerate(os.listdir(split_txt_dir)):
        txt_path = os.path.join(split_txt_dir, item)
        filename = data.rsplit(".mp4")[0]+"_"+str(i)
        filename = os.path.join(f"tmp/process", filename)
        os.makedirs(filename, exist_ok=True)
        split_parameters = []
        with open(txt_path) as f:
            for line in islice(f, 7, None):
                var = line.strip().split(" 	")
                split_parameters.append([float(row) for row in var])
        video = cv2.VideoCapture(data)
        video.set(1, int(split_parameters[0][0]))
        start = split_parameters[0][0]/25
        end = split_parameters[-1][0]/25
        print(start, end)
        # print(split_parameters[0])
        x1, x2, y1, y2 = 1920, 0, 1080, 0
        for j in range(len(split_parameters)):
            ret, frame = video.read()
            _, x, y, w, h = split_parameters[j]
            x, y, w, h = x*1920, y*1080, w*1920, h*1080
            n_x1, n_x2, n_y1, n_y2 = x, x+w, y-h*0.2, y+h
            x1 = math.floor(max(0, min(x1, n_x1)))
            x2 = math.ceil(min(1920-1, max(x2, n_x2)))
            y1 = math.floor(max(0, min(y1, n_y1)))
            y2 = math.ceil(min(1080-1, max(y2, n_y2)))
        video.release()
        video_path.append(filename)
        subprocess.call(f"ffmpeg -i {data} -ss {start} -to {end} -filter:v 'crop={x2-x1}:{y2-y1}:{x1}:{y1}'  {filename}/video.mp4",
                        shell=True, stdout=DEVNULL, stderr=DEVNULL)
        subprocess.call(f"ffmpeg -i  {data} -ss {start} -to {end} {filename}/audio.wav",
                        shell=True, stdout=DEVNULL, stderr=DEVNULL)
    return video_path


def get_landmark_bbox(data: np.ndarray, parameters: dict):
    max_h, max_w = data.shape[:-1]
    # print(len(fa.get_landmarks(data, return_bboxes=True)))
    # print(type(fa.get_landmarks(data, return_bboxes=True)))
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
        print(eye_dist)
        parameters["valid"] = False

        raise ValidException()
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
        bbox = parameters["bbox"]
        # offset = bbox[:2]
        # desiredLeftEye = (0.4, 0.4)
        rotate_center = parameters["landmark"][29]  # -offset
        # rotate_center=(parameters["landmark"][39]+parameters["landmark"][42])/2#-offset
        # cv2.circle(data, tuple(rotate_center.astype(np.int16)),
        #            1, (255, 255, 255), 1)
        M = cv2.getRotationMatrix2D(rotate_center, parameters["angle"], 1)
        h, w = data.shape[:-1]
        # fw, fh = bbox[2:]-bbox[:2]
        # tx = fw*0.5
        # ty = fh*0.4
        # print()
        # M[0, 2] += (tx-rotate_center[0])
        # M[1, 2] += (ty-rotate_center[1])
        data = cv2.warpAffine(data, M, (w, h))  # (int(fw), int(fh)))
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
    last_uri=''
    if not os.path.exists('tmp/video'):
        os.makedirs('tmp/video')
    for i, item in enumerate(job_list):
        try:
            parameters = item["parameters"]
            if not parameters["valid"]:
                continue
            if 'save_path' in parameters:
                yield parameters['save_path']
                continue
            if last_uri!='' and last_uri!=item["uri"]:
                print('remove')
                os.remove(raw_path)
            last_uri=item["uri"]
            raw_path = item["uri"]
            if "https://" in item["uri"]: 
                if not os.path.exists('tmp/video/'+item["uri"].split("watch?v=")[-1]+'.mp4'):
                    print(('tmp/video/'+item["uri"].split("watch?v=")[-1]+'.mp4'))
                    print("download")
                    raw_path,parameters = download(item["uri"], parameters)
                else:
                    raw_path='tmp/video/'+item["uri"].split("watch?v=")[-1]+'.mp4'
            video_dir = avspeech_preprocess(
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
                    # print(video_dir,frame_num)
                    if not ret:
                        break
                    # parameters = {"valid": True}
                    data, parameters = get_landmark_bbox(frame, parameters)
                    data, parameters = eye_dist(data, parameters)
                    data, parameters = get_angle(data, parameters)
                    # data, parameters = crop_data(data, parameters)
                    data, parameters = rotate_image(data, parameters)
                    data, parameters = crop_data(data, parameters)
                    landmark_list.append(parameters["landmark"])
                    bbox_list.append(parameters["bbox"])
                    angle_list.append(parameters["angle"])
                    save_path = os.path.join(
                        video_dir, str(frame_num).zfill(5)+".png")
                    cv2.imwrite(save_path, data)
                    frame_num += 1
                video.release()
                os.remove(video_path)
                np.savez(f"{video_dir}/data", landmark=landmark_list,
                            bbox=bbox_list, angle=angle_list)
                move_data(
                    video_dir, f"currect/{basename(video_dir)}")
                video_dir=f"currect/{basename(video_dir)}"
                yield video_dir
            except Exception as e:
                print(video_dir,"is not valid")
                shutil.rmtree(video_dir)
        except Exception as e:
            print("download error or split video error")


if __name__ == "__main__":
    job_list = [
        {"uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "parameters": {"id":1,"valid": True, "is_avspeech": True,
                        "timestamp": {"start": 28.84, "end": 35.16},
                        "bbox_crop": {'x': 0, 'y': 0, 'w': 1920, 'h': 1080},
                        "save_path": "currect:/E0NdymcK7wg_1"
                        }
        },
        {"uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "parameters": {"id":2,"valid": True, "is_avspeech": True,
                    "timestamp": {"start": 39, "end": 43.44},
                    "bbox_crop": {'x': 0, 'y': 0, 'w': 1920, 'h': 1080}
                    }
        },
        {"uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "parameters": {"id":3,"valid": True, "is_avspeech": True,
                    "timestamp":  {"start": 60.24, "end": 69.68},
                    "bbox_crop":{'x': 0, 'y': 0, 'w': 1920, 'h': 1080}
                    }
        },
        {"uri": "https://youtube.com/watch?v=sPJ365h2rxI",
            "parameters": {"id":1,"valid": True, "is_avspeech": True,
                    "timestamp":  {"start": 179.64, "end": 205.08},
                    "bbox_crop":{'x': 0, 'y': 0, 'w': 1920, 'h': 1080}
                    }
        },
    ]
    for item in main_pipeline(job_list):
        print("item", item)