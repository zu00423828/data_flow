import sqlite3
import pandas as pd
from sqlite3 import connect, Row
from typing import List, Tuple
from datetime import datetime
from uuid import uuid4
import os
from multiprocessing import Pool


class JobSession:
    def __init__(self, connection: sqlite3.Connection, limit=10, dataset_type=None, unfished_jobs=True):
        self.conn = connection
        self.cur = self.conn.cursor()
        self.cur.execute('PRAGMA foreign_keys = ON;')
        self.limit = limit
        self.processing_ticket_id: int = None
        self.tic = None
        self.dataset_type = dataset_type
        self.unfished_jobs = unfished_jobs

    def __enter__(self):
        insert_ticket = """INSERT INTO processing_ticket
        (value) VALUES (:value)
        """
        hex = uuid4().hex
        self.cur.execute(insert_ticket, {'value': hex})
        self.cur.execute("""SELECT id FROM processing_ticket WHERE value = :value""", {
            "value": hex,
        })
        self.conn.commit()
        self.processing_ticket_id = self.cur.fetchone()['id']

        where = """AND y.path IS NULL
                AND y.valid = true"""
        if self.unfished_jobs is None:
            where = ""
        elif not self.unfished_jobs:
            where = """AND y.path IS NOT NULL
                AND y.valid = true"""

        dataset_type = "" if self.dataset_type is None else f"AND d.value = '{self.dataset_type}'"

        script = """UPDATE uri SET processing_ticket_id = :processing_ticket_id
        WHERE id IN (
            SELECT uri.id FROM uri
            LEFT JOIN youtube_speech AS y
                ON y.uri_id = uri.id
            LEFT JOIN dataset_type AS d
                ON d.id = y.dataset_type_id
            WHERE
                processing_ticket_id IS NULL
                {where}
                {dataset_type}
            LIMIT :limit
        )
        """.format(where=where, dataset_type=dataset_type)

        self.cur.execute(script, {
            'processing_ticket_id': self.processing_ticket_id,
            'limit': self.limit,
        })
        self.conn.commit()

        self.tic = datetime.now()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print("got exit error:")
            print("type:", exc_type)
            print("exc_val:", exc_val)
            print("exc_tb:", exc_tb)
            self.conn.rollback()

        try:
            self.cur.execute("""DELETE FROM processing_ticket WHERE id = :id""",
                             {'id': self.processing_ticket_id})
            self.conn.commit()
        except Exception as err:
            self.conn.rollback()
            print("exit session failed: {}".format(err))
        print('[{}] Job Session Cost: {}'.format(
            self.processing_ticket_id, datetime.now() - self.tic))


class YoutubeSpeechDB:

    def __init__(self, path: str):
        self.conn = connect(path, timeout=100)
        self.conn.row_factory = Row
        self.cur = self.conn.cursor()
        self._init_tables()

    def __delete__(self, instance):
        self.conn.close()

    def _init_tables(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_type (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value TEXT NOT NULL UNIQUE
        )
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS processing_ticket (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value TEXT NOT NULL UNIQUE
        )
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS uri (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            value TEXT NOT NULL UNIQUE,
            processing_ticket_id INTEGER NULL,
            FOREIGN KEY(processing_ticket_id)
            REFERENCES processing_ticket(id)
            ON DELETE SET NULL ON update CASCADE
        )
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS youtube_speech (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_type_id INTEGER NOT NULL,
            split INTEGER NOT NULL,
            uri_id INTEGER NOT NULL,
            valid BOOLEAN NOT NULL DEFAULT 1,
            start REAL NOT NULL,
            end REAL NOT NULL,
            x0 INTEGER NOT NULL,
            y0 INTEGER NOT NULL,
            w INTEGER NOT NULL,
            h INTEGER NOT NULL,
            path TEXT NULL UNIQUE,
            landmarks BLOB NULL,
            bboxes BLOB NULL,
            angles BLOB NULL,
            FOREIGN KEY(uri_id) REFERENCES uri(id) ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY(dataset_type_id) REFERENCES dataset_type(id) ON DELETE CASCADE ON UPDATE CASCADE,
            UNIQUE(uri_id, split)
        )
        """)
        self.conn.commit()

    def update_job(self, youtube_speech_id: int, valid: bool, path: str = None,
                   landmarks=None, bboxes=None, angles=None):
        script = """
        UPDATE youtube_speech
        SET path = :path,
            valid = :valid,
            landmarks = :landmarks,
            bboxes = :bboxes,
            angles = :angles
        WHERE
            id = :youtube_speech_id
        """
        self.cur.execute(script, {
            'youtube_speech_id': youtube_speech_id,
            'valid': valid,
            'path': path,
            'landmarks': landmarks,
            'bboxes': bboxes,
            'angles': angles,
        })
        self.conn.commit()

    def add_job(self, uri_id: int, segments: List[Tuple[float, float]], dataset_type_id: int, autocommit=True):
        script = """
        INSERT INTO youtube_speech (dataset_type_id, split, uri_id, start, end, x0, y0, w, h)
        VALUES (:dataset_type_id, :split, :uri_id, :start, :end, :x0, :y0, :w, :h)
        """
        rows = []
        for i, segment in enumerate(segments):
            start, end, x0, y0, w, h = segment
            row = {
                'dataset_type_id': dataset_type_id,
                'uri_id': uri_id,
                'split': i,
                'start': start,
                'end': end,
                'x0': x0,
                'y0': y0,
                'w': w,
                'h': h,
            }
            rows.append(row)
        try:
            self.cur.executemany(script, rows)
            if autocommit:
                self.conn.commit()
        except Exception as err:
            print("Add job failed: {}".format(err))
            self.conn.rollback()

    def session(self, limit=10, dataset_type=None, unfished_jobs=True):
        return JobSession(self.conn, limit=limit, dataset_type=dataset_type, unfished_jobs=unfished_jobs)

    def list_jobs(self, processing_ticket_id: int) -> List[dict]:
        """
        Example:
        ```
        db = YoutubeSpeechDB(DB_PATH)
        with db.session(limit=15, dataset_type=None) as sess:
            jobs = db.list_jobs(processing_ticket_id=sess.processing_ticket_id)
            assert len(jobs) > 0
            print("[{}] do jobs".format(sess.processing_ticket_id))
            for job in jobs:
                db.update_job(job['id'], valid=False)
            print("[{}] release jobs".format(sess.processing_ticket_id))
        ```
        """

        script = """
        SELECT y.id,
            y.split,
            u.value AS uri,
            y.valid,
            y.start,
            y.end,
            y.x0,
            y.y0,
            y.w,
            y.h,
            y.path
        FROM youtube_speech AS y
        LEFT JOIN uri AS u
            ON u.id = y.uri_id
        LEFT JOIN processing_ticket AS p
            ON u.processing_ticket_id = p.id
        WHERE
            p.id = :processing_ticket_id
        """

        self.cur.execute(
            script, {'processing_ticket_id': processing_ticket_id})
        jobs = [dict(job) for job in self.cur.fetchall()]
        print("list {} jobs".format(len(jobs)))
        return jobs


def _collect_vox(job):
    filepath, youtube_id = job
    df = pd.read_csv(filepath, sep='\t', skiprows=7, header=None)
    df.columns = ['FRAME', 'X0', 'Y0', 'W', 'H']
    df['X1'] = df['X0'] + df['W']
    df['Y1'] = df['Y0'] + df['H']
    start = df.iloc[0]['FRAME'] / 25.0
    end = df.iloc[-1]['FRAME'] / 25.0
    x0 = int(df['X0'].min() * 1920)
    y0 = int(df['Y0'].min() * 1080)
    x1 = int(df['X1'].max() * 1920)
    y1 = int(df['Y1'].max() * 1080)
    w = x1 - x0
    h = y1 - y0
    row = {
        'YouTubeID': youtube_id,
        'start_segment': start,
        'end_segment': end,
        'X0': x0,
        'Y0': y0,
        'W': w,
        'H': h,
    }
    return row

class YoutubeSpeech:

    def __init__(self, df: pd.DataFrame):
        required_columns = [
            'YouTubeID', 'start_segment',
            'end_segment', 'X0', 'Y0', 'W', 'H']
        self.df = df
        for col in required_columns:
            assert col in self.df.columns, f"column {col} not found"
        self.video_splits = {}
        self._arrange_same_video()

    def _arrange_same_video(self):
        print("arrange same video")
        for _, row in self.df.iterrows():
            id = row['YouTubeID']
            start = row['start_segment']
            end = row['end_segment']
            x0 = row['X0']
            y0 = row['Y0']
            w = row['W']
            h = row['H']
            if id not in self.video_splits:
                self.video_splits[id] = [
                    (start, end, x0, y0, w, h)
                ]
                continue
            self.video_splits[id].append((start, end, x0, y0, w, h))

        for video_id in self.video_splits.keys():
            segments = self.video_splits[video_id]
            self.video_splits[video_id] = sorted(
                segments, key=lambda s: s[0])

    @classmethod
    def from_avspeech(cls, csv: str):
        df = pd.read_csv(csv, header=None)
        df.columns = [
            'YouTubeID', 'start_segment',
            'end_segment', 'X_center', 'Y_center']

        # fix full screen
        df['W'] = 1920
        df['H'] = 1080
        df['X0'] = 0
        df['Y0'] = 0
        return cls(df)

    @classmethod
    def from_vox(cls, dirpath: str):
        rows = []
        txt_dirpath = os.path.join(dirpath, 'txt')
        if os.path.exists(txt_dirpath):
            dirpath = txt_dirpath

        jobs = []
        for id_dir in os.listdir(dirpath):
            id_dirpath = os.path.join(dirpath, id_dir)
            for youtube_id in os.listdir(id_dirpath):
                youtube_dirpath = os.path.join(id_dirpath, youtube_id)
                for filename in os.listdir(youtube_dirpath):
                    filepath = os.path.join(youtube_dirpath, filename)
                    job = filepath, youtube_id
                    jobs.append(job)
        rows = []
        for i, row in enumerate(Pool().imap_unordered(_collect_vox, jobs)):
            if i % 10000 == 0:
                print("collect {} rows".format(i))
            rows.append(row)
        
        return cls(pd.DataFrame(rows))

    def to_db(self, path: str, dataset_type: str):
        insert_uri = """
        INSERT INTO uri (value)
        VALUES (:value)
        """
        find_uri = """
        SELECT id FROM uri
        WHERE value = :value
        """
        print("insert data to db:", path)
        db = YoutubeSpeechDB(path)

        db.cur.execute("""INSERT OR IGNORE INTO dataset_type
        (value) VALUES (:value)
        """, {
            'value': dataset_type,
        })
        db.conn.commit()
        db.cur.execute("""SELECT id FROM dataset_type WHERE value = :value""", {
            'value': dataset_type,
        })
        dataset_type_id = db.cur.fetchone()['id']

        i = 0
        for video_id, segments in self.video_splits.items():
            uri = f'https://youtube.com/watch?v={video_id}'
            db.cur.execute(find_uri, {'value': uri})
            found = db.cur.fetchone()
            if found is not None:
                print(f"[bypass] uri already exists: {uri}")
                continue
            db.cur.execute(insert_uri, {'value': uri})
            db.cur.execute(find_uri, {'value': uri})
            uri_id = db.cur.fetchone()['id']
            db.add_job(uri_id, segments, dataset_type_id=dataset_type_id, autocommit=False)
            i += 1

            if i % 10240 == 0:
                print(f"committed {i} videos")
                db.conn.commit()
        print(f"committed {i} rows")
        db.conn.commit()
