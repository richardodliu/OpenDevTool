import openai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
from loguru import logger
import time
import random
from openai import OpenAI

import time
import random

def retry(max_retries=3, initial_delay=2, backoff_factor=2, jitter=0.5, exceptions=(openai.APIStatusError,)):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    print(f"Retry {retries + 1}/{max_retries} for function {func.__name__} failed with error: {e}")
                    time.sleep(delay)
                    retries += 1
                    # 使用随机扰动来增加延迟时间的随机性
                    random_jitter = random.uniform(1 - jitter, 1 + jitter)
                    delay = min(delay * backoff_factor, delay * random_jitter)
            return None  # or raise an exception if you prefer
        return wrapper
    return decorator

client = OpenAI(
    api_key="sk-",
    base_url="https://api.deepseek.com"
)


@retry(max_retries=5, exceptions=(openai.APIStatusError,))
def process_line(line):
    msg = [
        {"role": "system", "content": "You are a very helpful assistant."},
        {"role": "user", "content": "请开始你的表演"}
    ]
    data = json.loads(line)
    text = data['text'][:500]

    PROMPT= """
    <begin of text>{text}<end of text> 我们正在开发一个知识库项目，从互联网上获取和wiki内容相似的高质量数据。因此，需要针对上面<begin of text>和<end of text>之间的文本进行深入研究，这些信息将用于构建一个类似于维基百科的知识平台。我们现在收集到了很多网页，但其中混杂着新闻网页、游戏相关网页、小说相关网页、广告相关网页。我希望你承担一个分类器的角色，仔细阅读上面<begin of text>和<end of text>之间的文本,并将该文本分为[新闻，游戏，小说，广告，知识]这五个类别之一。你输出的结果必须遵循我给出的格式进行输出：请输出[新闻，游戏，小说，广告，知识]中的类别之一，比如如果该文本是移民相关广告或学校招生广告，则输出“广告”。当无法判断时，请输出“无法判断”。
    """.strip()

    prompt = PROMPT.format(text=text)
    msg[1]['content'] = prompt

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=msg,
        stream=False
    )
    if response and response.choices:
        content = response.choices[0].message.content
        data['game_classifier'] = content
        logger.debug(f"[INFO] result: {content}")
        return json.dumps(data, ensure_ascii=False)
    else:
        logger.debug(f"[INFO] response: {response}")
    # except openai.APIStatusError as e:
        # print(f"[ERROR] {str(datetime.datetime.now())} headers:{e.response.headers}, resp:{e.response.content}")

jsonl_path = '1.jsonl'
output_jsonl_path = '2.jsonl'  # 输出文件路径
lines = []

with open(jsonl_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

random.shuffle(lines)
# lines = lines[:1000]
logger.debug(f"[DEBUG] load lines sucess, all lines: {len(lines)}")
# 使用线程池处理每行数据
with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(process_line, line): line for line in lines}
    with open(output_jsonl_path, 'a', encoding='utf-8') as outfile:
        for future in as_completed(futures):
            result = future.result()
            if result:
                outfile.write(result + '\n')  # 将更新后的数据写入到新的jsonl文件中

print(f"[INFO] finish!!!")