## 使用方法

```
from dataflow import main_pipeline 
```

## 使用範例
```
    job_list = [
        {"uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "parameters": {"id":1,"valid": True, "is_avspeech": True,
                        "timestamp": {"start": 28.84, "end": 35.16},
                        "bbox_crop": {'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080},
                        # "save_path": "currect/E0NdymcK7wg_1"
                        }
        },
        {"uri": "https://youtube.com/watch?v=E0NdymcK7wg",
            "parameters": {"id":2,"valid": True, "is_avspeech": True,
                    "timestamp": {"start": 39, "end": 43.44},
                    "bbox_crop": {'x0': 0, 'y0': 0, 'w': 1920, 'h': 1080}
                    }
        },
    ]
    main_pipeline(jobs_list)
```

## 參數說明
- uri為 下載的網址
- parameter為參數
    - valid 代表可用
    - timestamp 分割的時間
        - start 開始時間
        - end 結束時間
    - bbox_crop 切割影片大小的方框
        - x0 方框左上角
        - y0 方框右上角
        - w 方框寬度
        - h 方框高度
    - save_path 為最後儲存的資料夾路徑(包含 png 跟 npz以及 wav檔)

## 輸出parameters
- landmark  影片切割後的各幀的landmark(還沒校正)
- bbox   影片切割後的各幀的bbox(還沒校正)
- angle  影片切割後的各幀的人臉角度(還沒校正)
