from dataflow import main_pipeline
import tensorflow as tf
import cv2
import glob
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_generater():
    job_list = [
        {"uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "parameters": {"id":1,"valid": True, "is_avspeech": True,
                    "timestamp": {"start": 28.84, "end": 35.16},
                    "bbox_crop": {'x': 0, 'y': 0, 'w': 1920, 'h': 1080},
                    "save_path":"currect/E0NdymcK7wg_1"
                    }
        },
        {"uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "parameters": {"id":2,"valid": True, "is_avspeech": True,
                    "timestamp": {"start": 39, "end": 43.44},
                    "bbox_crop": {'x': 0, 'y': 0, 'w': 1920, 'h': 1080},
                    "save_path":"currect/E0NdymcK7wg_2"
                    }
        },
        {"uri": "https://youtube.com/watch?v=sPJ365h2rxI",
            "parameters": {"id":1,"valid": True, "is_avspeech": True,
                    "timestamp":  {"start": 179.64, "end": 205.08},
                    "bbox_crop":{'x': 0, 'y': 0, 'w': 1920, 'h': 1080}
                    }
        },
    ]

    for dir in main_pipeline(job_list):
        frame_list=glob.glob(f'{dir}/*.png')
        frame_num=len(frame_list)
        ridx=np.random.choice(frame_num,replace=True,size=2)
        source=cv2.imread(frame_list[ridx[0]])
        driving=cv2.imread(frame_list[ridx[1]])
        source=cv2.resize(source,(256,256))
        driving=cv2.resize(driving,(256,256))
        yield source,driving
        # for item in glob.glob(f'{dir}/*.png'):
            # img=cv2.imread(item)
            # img=cv2.resize(img,(256,256))
            # yield img
# for item in test_generater():
#     print("item",item)
generator=tf.data.Dataset.from_generator(test_generater,output_types=(tf.uint8,tf.uint8),output_shapes=((256,256,3),(256,256,3)))
for source,driving in generator:
    print(type(source),type(driving))
    print(source.shape,driving.shape)

