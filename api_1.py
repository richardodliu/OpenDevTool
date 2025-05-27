import openai
import json
import random
import os
import time
import copy
from multiprocessing import Pool
import hashlib
import tiktoken

# 设置
PROMPT = """Assess if a URL indicates it may lead to content that's 'wiki-like', defined broadly as webpages aimed at explaining or demonstrating specific concepts or subjects, systematically organized and written to educate. Look for structure in the URL or keywords suggesting educational, systematic, or explanatory content. Answer only yes or no.
URL: <URL>"""
MAX_RANDOM_BACKOFF = 5
OUTPUT_DIR = 'gpt_filtered'
FAILED_DIR = os.path.join(OUTPUT_DIR, "failed")
TIMEOUT = 300
CHUNK_SIZE = 1024
NUM_PROCESSES = 200
os.makedirs(FAILED_DIR, exist_ok=True)

RESULT_FILE = os.path.join(OUTPUT_DIR, "results.jsonl")
stream = False
# 自定义API设置
model = "gpt-4-1106-preview"
openai.base_url = ''
# 不要替换这个 无实际意义, 但是openai sdk不允许这个值为空, 所以随便设置一个值就行
openai.api_key = 'no-modify' # 不是改这个值，改下面的
# Helper functions



def save_json(json_data, path):
    assert isinstance(json_data, dict), "json_data must be a dict"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_log(text, path):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text+"\n")

def save_raw(text, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def hash_string(string):
    return hashlib.md5(string.encode()).hexdigest()

def save_res(file, result, data):
    saved_res = copy.deepcopy(data)
    data['result'] = result
    with open(file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    print(f"[INFO] save!")

def call(data, retry=1):

    try:
        if MAX_RANDOM_BACKOFF > 0:
            time.sleep(random.uniform(0, MAX_RANDOM_BACKOFF))

        full_prompt = data["prompt"]
        # print(full_prompt)

        print("before")
        # 使用自定义API端点和额外的头部信息
        # results = openai.ChatCompletion.create(
        #     model='gpt-4-1106-preview',
        #     messages=[
        #         {"role": "user", "content": full_prompt},
        #     ],
        #     extra_headers={'apikey': 'sk-'},
        #     stream = stream
        # )["choices"][0]["message"]["content"]
        # print(full_prompt)
        msg = [
        {"role": "system", "content": "You are a vary helpful assistant."},
        {"role": "user", "content": full_prompt},
    ]

        results = openai.chat.completions.create(
        model=model,
        messages=msg,
        extra_headers={'apikey':'sk'},
        stream=stream).choices[0].message.content

        print(results)
        try:
            # save result
            save_res(RESULT_FILE, results, data)
        except Exception as e:
            error_uid = hash_string(str(results))
            save_raw(results, os.path.join(FAILED_DIR, error_uid + ".txt"))
            print(f"Failed: format error {error_uid}")
    except Exception as e:
        save_log(f"{str(e)}\n", os.path.join(OUTPUT_DIR, "failed_log.txt"))
        # print(f"URL {data['sub_url']} Failed: other error, retrying: {retry}")
        if retry <= 3:  # 重试上限设置为3次
            call(data, retry + 1)

def data_generator(chunk_size):
    items = []
    input_data = load_json('remaining_candidates.json')
    for root_url, sub_urls in input_data.items():