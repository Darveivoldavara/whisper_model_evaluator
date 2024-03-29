import httpx
from num2words import num2words
import re
import zipfile
import urllib.request
import json
import websockets
import requests
# from google.cloud import speech_v1p1beta1
import io
import os
import time
import jiwer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def transcribe_yandex(filelink, key):    

    POST = "https://transcribe.api.cloud.yandex.net/speech/stt/v2/longRunningRecognize"

    body ={
        "config": {
            "specification": {
                "languageCode": "ru-RU"
            }
        },
        "audio": {
            "uri": filelink
        }
    }

    # Если вы хотите использовать IAM-токен для аутентификации, замените Api-Key на Bearer.
    header = {'Authorization': 'Api-Key {}'.format(key)}

    # Отправить запрос на распознавание.
    req = requests.post(POST, headers=header, json=body)
    data = req.json()
    print(data)

    id = data['id']

    # Запрашивать на сервере статус операции, пока распознавание не будет завершено.
    while True:

        time.sleep(1)

        GET = "https://operation.api.cloud.yandex.net/operations/{id}"
        req = requests.get(GET.format(id=id), headers=header)
        req = req.json()

        if req['done']: break
        #print("Not ready")

    # Показать полный ответ сервера в формате JSON.
    #print("Response:")
    #print(json.dumps(req, ensure_ascii=False, indent=2))

    # Показать только текст из результатов распознавания.
    #print("Text chunks:")
    result = ''
    for chunk in req['response']['chunks']:
        result += chunk['alternatives'][0]['text'] + ' '

    return result.lower()


def transcribe_sova(url, file_name):
    
    # https://habr.com/ru/company/ashmanov_net/blog/523412/
    # https://github.com/sovaai/sova-asr
    
    with open(file_name, 'rb') as f1:
        response = requests.post(url,files={"audio_blob": f1}).text
    return json.loads(response)['r'][0]['response'][0]['text']


def transcribe_google(file_path):
    
    # https://cloud.google.com/speech-to-text
    # https://cloud.google.com/speech-to-text/docs/reference/rest/v1/RecognitionConfig?hl=ru#AudioEncoding
        
    language = 'ru'
    client = speech_v1p1beta1.SpeechClient()    
    sample_rate_hertz = 8000
    encoding = speech_v1p1beta1.RecognitionConfig.AudioEncoding.MP3

    config = {
        "language_code": language,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
    }
    with io.open(file_path, 'rb') as audio_file:
        content = audio_file.read()
        audio = speech_v1p1beta1.RecognitionAudio(content=content)

        response = client.recognize(config = config, audio = audio)
        results = []
        for result in response.results:
            alternative = result.alternatives[0]
            results.append(alternative.transcript)
    return ''.join([str(item) for item in results]).lower()


async def transcribe_vosk(file_path, server):
    async with websockets.connect(server) as websocket:
        
        phrases = []
        with open(file_path, "rb") as audio_file:
            while True:
                data = audio_file.read(8000)
                if len(data) == 0:
                    break
                await websocket.send(data)
                accept = json.loads(await websocket.recv())
                if len(accept)>1 and accept['text'] != '':
                    accept_text = str(accept['text'])
                    phrases.append(accept_text)

            await websocket.send('{"eof" : 1}')
            accept = json.loads(await websocket.recv())
            if len(accept)>1 and accept['text'] != '':
                accept_text = str(accept['text'])
                phrases.append(accept_text)

    return phrases


"""def async_vosk(file_path, server):
    
    phrases = asyncio.get_event_loop().run_until_complete(transcribe_vosk(file_path, server))
    return phrases
"""


def clean_text(text):
    text = text.replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\b\d+\b", lambda x: num2words(int(x.group()), lang="ru"), text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


async def transcribe_whisper(file_path, server_url, vad=1):

    # https://github.com/linto-ai/whisper-timestamped
    # https://github.com/openai/whisper

    phrases = []
    async with httpx.AsyncClient(timeout=None) as client:
        files = {
            "file": (os.path.basename(file_path), open(file_path, "rb"), "audio/wav")
        }
        data = {"vad": vad}

        response = await client.post(server_url, files=files, data=data)
        if response.status_code == 200:
            data = response.json()
            if "segments" in data:
                for segment in data["segments"]:
                    cleaned_text = clean_text(segment["text"])
                    phrases.append(cleaned_text)
            else:
                cleaned_text = clean_text(data["text"])
                phrases.append(cleaned_text)
        else:
            print(f"Error processing file {file_path}: {response.text}")

    return phrases


def error(ground_truth, hypothesis):
    
    wer = jiwer.wer(ground_truth, hypothesis)
    mer = jiwer.mer(ground_truth, hypothesis)
    wil = jiwer.wil(ground_truth, hypothesis)

    # faster, because `compute_measures` only needs to perform the heavy lifting once:
    measures = jiwer.compute_measures(ground_truth, hypothesis)
    return measures


def load_data(files):
    
    frames = [pd.read_csv(file) for file in files]
    return pd.concat(frames, ignore_index=True)


def dictonary_init_print(path, skip_init = False):
    if not skip_init:
        print("human = []")
    for root, dirs, files in os.walk(path):
        files.sort()
        for filename in files:
            print("human.append({'path': '" + path + "', 'file': '" + filename + "', 'human_text': ''})")


def evaluate(df, name):
    
    wer = []
    mer = []
    wil = []
    for row in range(len(df)):
        measures = error(df.iloc[row].human_text, df.iloc[row][name + '_text'])
        wer.append(measures['wer'])
        mer.append(measures['mer'])
        wil.append(measures['wil'])
    df[name + '_wer'] = wer
    df[name + '_mer'] = mer
    df[name + '_wil'] = wil


def plot(df, names):

    errors = ['wer', 'mer', 'wil']
    for r in errors:
        df.plot(y=[n + "_" + r for n in names], kind="bar", width=0.9, figsize=(20, 6))
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
        plt.grid(linestyle='--', alpha=0.5)
        plt.show()


def comparator(df, engines, evals):

    comparing = []
    for g in engines:
        for v in evals:
            comparing.append({
                'engine': g,
                'eval_func': v,
                'val': np.median(df[g + '_' + v])
            })
    comp_df = pd.DataFrame(comparing)

    plt.figure(figsize=(20, 7))

    for v in evals:
        data_pointer = comp_df[comp_df.eval_func == v]
        plt.plot(data_pointer.engine, data_pointer.val, label=v, linewidth=3)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
    plt.grid(linestyle='--', alpha=0.5)
    plt.show()

    for v in evals:
        print('\n' + v + ':')
        for g in engines:    
            print(g, np.median(df[g + '_' + v]))


def examples(df, engines, limit = 0):

    for i in range(min(len(df), len(df) if limit == 0 else limit)):

        print('\n=== example ' + str(i) + ' ===')

        for g in engines:
            print('\n' + g + ':')
            print(df.loc[i][g + '_text'])


def download_dataset(URL, data_path, archive_file_name):

    print('Downloading...')
    urllib.request.urlretrieve(URL, archive_file_name)
    print('Extracting zip...')
    with zipfile.ZipFile(archive_file_name, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print('Dataset is ready!')
