import datetime
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.text import Tokenizer

from constants import (
    AGE_CLASS,
    CITY_CLASS,
    CSV_PATH,
    DROP,
    EMPLOYMENT_CLASS,
    EXPERIENCE_CLASS,
    FILTERS,
    NUM_WORDS,
    SCHEDULE_CLASS,
    URL
)


def download_file(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with streaming to avoid keeping it all in memory."""
    if dst.exists() and dst.stat().st_size > 0:
        return

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


download_file(URL, CSV_PATH)

df = pd.read_csv(
    CSV_PATH,
    delimiter='|',
    on_bad_lines='skip',
    low_memory=False,
    index_col=0
)

df = df.drop(columns=DROP, errors='ignore')
df = df[df['salary'].notna()]
df = df[df['positionName'].notna()]


def load_json(js):
    """Safely parse JSON stored as a string; return [] on errors."""
    try:
        return json.loads(js)
    except (TypeError, json.JSONDecodeError):
        return []


df['workExperienceList'] = df['workExperienceList'].apply(load_json)
df['educationList'] = df['educationList'].apply(load_json)

COL_LOCALITY = df.columns.get_loc('localityName')
COL_EXPERIENCE = df.columns.get_loc('experience')
COL_AGE = df.columns.get_loc('age')
COL_BUSY = df.columns.get_loc('busyType')
COL_SCHED = df.columns.get_loc('scheduleType')
COL_SALARY = df.columns.get_loc('salary')


def city20HE(param):
    """One-hot encode city class from a string."""
    num_classes = len(set(CITY_CLASS.values()))

    if not isinstance(param, str):
        param = list(CITY_CLASS.keys())[-1]

    split_array = re.split(r'[ ,.:()?!]+', param)
    city_class = -1

    for word in split_array:
        city_class = CITY_CLASS.get(word, -1)
        if city_class >= 0:
            break
    else:
        city_class = num_classes - 1
    return utils.to_categorical(city_class, num_classes)


def range20HE(param, class_list):
    """Bucketize numeric value by thresholds and one-hot encode."""
    num_classes = len(class_list) + 1

    for i in range(num_classes - 1):
        if float(param) < class_list[i]:
            cls = i
            break
    else:
        cls = num_classes - 1
    return utils.to_categorical(cls, num_classes)


def str2OHE(param, class_dict):
    """One-hot encode a single categorical string with fallback."""
    num_classes = len(set(class_dict.values()))
    result = np.zeros(num_classes, dtype=np.float32)

    if not isinstance(param, str):
        param = list(class_dict.keys())[-1]
    else:
        param = param.strip()

    cls = class_dict.get(param, class_dict[list(class_dict.keys())[-1]])
    result[cls] = 1.0
    return result


def get_row_data(row):
    """Build (x, y) for a single row."""
    x_data = np.hstack([
        city20HE(row[COL_LOCALITY]),
        range20HE(row[COL_EXPERIENCE], EXPERIENCE_CLASS),
        range20HE(row[COL_AGE], AGE_CLASS),
        str2OHE(row[COL_BUSY], EMPLOYMENT_CLASS),
        str2OHE(row[COL_SCHED], SCHEDULE_CLASS)
    ]).astype(np.float32)
    y_data = np.array([row[COL_SALARY]]) / 1000
    return x_data, y_data


def get_train_data(dataFrame):
    """Convert dataframe rows to X/Y numpy arrays (tabular part)."""
    x_data = []
    y_data = []

    for row in dataFrame.values:
        x, y = get_row_data(row)
        x_data.append(x)
        y_data.append(y)
    return np.array(x_data), np.array(y_data)


x_train, y_train = get_train_data(df)


def extract_education(param):
    """Extract a single text string from educationList JSON."""
    edu_text = []

    if not isinstance(param, list):
        return ''

    for edu in param:
        if not isinstance(edu, dict):
            continue

        if edu.get('instituteName'):
            edu_text.append(edu['instituteName'])
        if edu.get('qualification'):
            edu_text.append(edu['qualification'])
        if edu.get('specialty'):
            edu_text.append(edu['specialty'])
        if edu.get('faculty'):
            edu_text.append(edu['faculty'])
        if edu.get('graduateYear'):
            edu_text.append(str(edu['graduateYear']))
    return '. '.join(edu_text)


def extract_works(param):
    """Extract a single text string from workExperienceList JSON."""
    work_text = []

    if not isinstance(param, list):
        return ''

    for job in param:
        if not isinstance(job, dict):
            continue

        if job.get('companyName'):
            work_text.append(job['companyName'])

        if job.get('dateFrom') and job.get('dateTo'):
            try:
                dateT = datetime.datetime.strptime(
                    job['dateTo'], '%Y-%m-%dT%H:%M:%S%z'
                )
                dateF = datetime.datetime.strptime(
                    job['dateFrom'], '%Y-%m-%dT%H:%M:%S%z'
                )
                months = int((dateT - dateF).total_seconds() / 2628000)
                work_text.append(f'Стаж {months} месяцев')
            except ValueError:
                pass

        if job.get('jobTitle'):
            work_text.append(job['jobTitle'])

        if job.get('achievements'):
            work_text.append(job['achievements'])

        if job.get('demands'):
            work_text.append(
                job['demands'].replace('<p>', '').replace('</p>', '')
            )
    return '. '.join(work_text)


df['education'] = df['educationList'].apply(extract_education)
df['works'] = df['workExperienceList'].apply(extract_works)
df['position'] = df['positionName'].astype(str)


def build_bow_matrix(texts, num_words):
    """Fit a tokenizer on texts and return a BoW matrix."""
    tok = Tokenizer(
        num_words=num_words,
        filters=FILTERS,
        lower=True,
        split=' ',
        oov_token='unknown',
        char_level=False,
    )
    tok.fit_on_texts(texts)
    seq = tok.texts_to_sequences(texts)
    return tok.sequences_to_matrix(seq).astype(np.float32)


x_train_education = build_bow_matrix(df['education'], NUM_WORDS)
x_train_works = build_bow_matrix(df['works'], NUM_WORDS)
x_train_position = build_bow_matrix(df['position'], NUM_WORDS)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
