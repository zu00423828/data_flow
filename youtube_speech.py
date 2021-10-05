import pandas as pd
from mysql.connector import pooling
# 千萬不要第一個import 它 ，否則會<urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1091)>
from typing import List, Tuple
from datetime import datetime
from uuid import uuid4
import os
from multiprocessing import Pool
from time import sleep


class JobSession:
    def __init__(self, conn: pooling.PooledMySQLConnection, limit=10, dataset_type=None, retries=3):
        self.conn: pooling.PooledMySQLConnection = conn
        self.limit = limit
        self.processing_ticket_id: int = None
        self.tic = None
        self.dataset_type = dataset_type
        self.retries = retries

    def __enter__(self):
        cur = self.conn.cursor(dictionary=True)
        insert_ticket = """INSERT INTO processing_ticket
        (value) VALUES (%(value)s)
        """
        for _ in range(self.retries):
            if self.processing_ticket_id is not None:
                cur = self.conn.cursor(dictionary=True)
                cur.execute("""DELETE FROM processing_ticket WHERE id = %(id)s""",
                            {'id': self.processing_ticket_id})
                self.conn.commit()
                self.processing_ticket_id = None

            hex = uuid4().hex
            cur.execute(insert_ticket, {'value': hex})
            self.conn.commit()
            cur.execute("""SELECT id FROM processing_ticket WHERE value = %(value)s""", {
                "value": hex,
            })
            self.processing_ticket_id = cur.fetchone()['id']

            where = """AND y.path IS NULL
                    AND y.valid = true"""

            dataset_type = "" if self.dataset_type is None else f"AND d.value = '{self.dataset_type}'"

            subquery = """SELECT DISTINCT(uri.id) AS id FROM uri
                LEFT JOIN youtube_speech AS y
                    ON y.uri_id = uri.id
                LEFT JOIN dataset_type AS d
                    ON d.id = y.dataset_type_id
                WHERE
                    processing_ticket_id IS NULL
                    {where}
                    {dataset_type}
                LIMIT %(limit)s""".format(where=where, dataset_type=dataset_type)

            cur.execute(subquery, {
                'limit': self.limit,
            })
            ids = [row['id'] for row in cur.fetchall()]

            script = """UPDATE uri
            SET processing_ticket_id = %(processing_ticket_id)s
            WHERE processing_ticket_id IS NULL
                AND uri.id IN ({the_ids})
            """.format(
                the_ids=", ".join([f"%(id_{i+1})s" for i in range(len(ids))]),
            )

            params = {
                'processing_ticket_id': self.processing_ticket_id,
            }
            for i, id in enumerate(ids):
                params[f"id_{i+1}"] = id

            cur.execute(script, params)

            self.conn.commit()
            if cur.rowcount > 0:
                break
            sleep(0.5)
        cur.close()
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
            cur = self.conn.cursor(dictionary=True)
            cur.execute("""DELETE FROM processing_ticket WHERE id = %(id)s""",
                        {'id': self.processing_ticket_id})
            self.conn.commit()
            cur.close()
        except Exception as err:
            self.conn.rollback()
            print("exit session failed: {}".format(err))
        print('[{}] Job Session Cost: {}'.format(
            self.processing_ticket_id, datetime.now() - self.tic))


class YoutubeSpeechDB:

    def __init__(self, ip_address: str, database='speech'):
        self.pool = pooling.MySQLConnectionPool(
            pool_name='pool',
            pool_reset_session=True,
            pool_size=2,
            user="root",
            password="root",
            host=ip_address,
            port="3456",
            database=database,
            charset='utf8mb4',
        )
        self.conn = self.pool.get_connection()
        self.cur = self.conn.cursor(dictionary=True)
        self._init_tables()

    def __delete__(self, instance):
        self.conn.close()

    def _init_tables(self):
        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS `dataset_type` (
            id TINYINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
            value TEXT NOT NULL,
            UNIQUE KEY `value` (value(255))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS `processing_ticket` (
            id MEDIUMINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
            value CHAR(32) NOT NULL,
            UNIQUE KEY `value` (value(32))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS `uri` (
            id MEDIUMINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
            value TEXT NOT NULL,
            processing_ticket_id MEDIUMINT UNSIGNED NULL,
            FOREIGN KEY(processing_ticket_id)
            REFERENCES processing_ticket(id)
            ON DELETE SET NULL ON update CASCADE,
            UNIQUE KEY `value` (value(255))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)

        self.cur.execute("""
        CREATE TABLE IF NOT EXISTS `youtube_speech` (
            id MEDIUMINT UNSIGNED PRIMARY KEY AUTO_INCREMENT,
            dataset_type_id TINYINT UNSIGNED NOT NULL,
            split TINYINT UNSIGNED NOT NULL,
            uri_id MEDIUMINT UNSIGNED NOT NULL,
            valid BOOLEAN NOT NULL DEFAULT true,
            start FLOAT NOT NULL,
            end FLOAT NOT NULL,
            x0 SMALLINT UNSIGNED NOT NULL,
            y0 SMALLINT UNSIGNED NOT NULL,
            w SMALLINT UNSIGNED NOT NULL,
            h SMALLINT UNSIGNED NOT NULL,
            path TEXT NULL,
            landmarks MEDIUMBLOB NULL,
            bboxes BLOB NULL,
            angles BLOB NULL,
            mel MEDIUMBLOB NULL,
            fps FLOAT NULL,
            frame_count SMALLINT UNSIGNED NULL,
            shift_frames TINYINT UNSIGNED NULL,
            FOREIGN KEY(uri_id) REFERENCES uri(id) ON DELETE CASCADE ON UPDATE CASCADE,
            FOREIGN KEY(dataset_type_id) REFERENCES dataset_type(id) ON DELETE CASCADE ON UPDATE CASCADE,
            UNIQUE KEY `uri_split_id` (`uri_id`, `split`),
            UNIQUE KEY `path` (path(255))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        self.conn.commit()

    def update_job(self, youtube_speech_id: int, valid: bool, path: str = None,
                   landmarks=None, bboxes=None, angles=None, fps=None, frame_count=None):
        script = """
        UPDATE youtube_speech
        SET path = %(path)s,
            valid = %(valid)s,
            landmarks = %(landmarks)s,
            bboxes = %(bboxes)s,
            angles = %(angles)s,
            fps = %(fps)s,
            frame_count = %(frame_count)s
        WHERE
            id = %(youtube_speech_id)s
        """
        self.cur.execute(script, {
            'youtube_speech_id': youtube_speech_id,
            'valid': valid,
            'path': path,
            'landmarks': landmarks,
            'bboxes': bboxes,
            'angles': angles,
            'fps': fps,
            'frame_count': frame_count,
        })
        self.conn.commit()

    def add_job(self, uri_id: int, segments: List[Tuple[float, float]], dataset_type_id: int, autocommit=True):
        script = """
        INSERT INTO youtube_speech (dataset_type_id, split, uri_id, start, end, x0, y0, w, h)
        VALUES (%(dataset_type_id)s, %(split)s, %(uri_id)s, %(start)s, %(end)s, %(x0)s, %(y0)s, %(w)s, %(h)s)
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

    def session(self, limit=10, dataset_type=None):
        return JobSession(self.conn, limit=limit, dataset_type=dataset_type)

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
            p.id = %(processing_ticket_id)s
            AND y.path IS NULL
            AND y.valid = true
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
    df['Y1'] = (df['Y0'] + df['H'])
    df['Y0'] = (df['Y0'] - df['H'] * 0.1)
    df['X1'] = (df['X0'] + df['W'])
    start = df.iloc[0]['FRAME'] / 25.0
    end = df.iloc[-1]['FRAME'] / 25.0
    x0 = max(int(df['X0'].min() * 1920), 0)
    y0 = max(int(df['Y0'].min() * 1080), 0)
    x1 = min(int(df['X1'].max() * 1920), 1920)
    y1 = min(int(df['Y1'].max() * 1080), 1080)
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

    def to_db(self, ip_address: str, dataset_type: str, database='speech'):
        insert_uri = """
        INSERT INTO uri (value)
        VALUES (%(value)s)
        """
        find_uri = """
        SELECT id FROM uri
        WHERE value = %(value)s
        """
        print("insert data to db:", ip_address)
        db = YoutubeSpeechDB(ip_address, database=database)

        db.cur.execute("""SELECT id FROM dataset_type
        WHERE value = %(value)s
        """, {
            'value': dataset_type,
        })
        found = db.cur.fetchone()
        if found is not None:
            dataset_type_id = found['id']
        else:
            db.cur.execute("""INSERT INTO dataset_type
            (value) VALUES (%(value)s)
            """, {
                'value': dataset_type,
            })
            db.conn.commit()
            db.cur.execute("""SELECT id FROM dataset_type WHERE value = %(value)s""", {
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
            db.add_job(uri_id, segments,
                       dataset_type_id=dataset_type_id, autocommit=False)
            i += 1

            if i % 10240 == 0:
                print(f"committed {i} videos")
                db.conn.commit()
        print(f"committed {i} rows")
        db.conn.commit()
