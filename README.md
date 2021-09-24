



## 使用範例
```
    from dataflow import main_pipeline 

    shareroot='/run/user/1000/gvfs/smb-share:server=192.168.10.25,share=shared/youtube-speech/'
    db_path=shareroot+'youtube_speech.sqlite'
    download_path=shareroot+'tmp/'
    tmp_path=shareroot+'tmp/video/'
    correct_path=shareroot+'correct/'

    for item in main_pipeline(db_path,download_path,tmp_path,correct_path):
        pass
```

