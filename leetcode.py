from demjson3 import decode
from decimal import Decimal
from tqdm import tqdm


def convert_decimals_to_float(data):
    """
    递归将数据结构中的所有 Decimal 类型转换为 float。
    """
    if isinstance(data, list):
        return [convert_decimals_to_float(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_decimals_to_float(value) for key, value in data.items()}
    elif isinstance(data, Decimal):
        return float(data)
    else:
        return data

def map_solution(example):
    try:
        example['meta'] = decode(example['json_'])
        # example['meta'] = convert_decimals_to_float(example['meta'])
        return example
    except Exception:
        example['meta'] = {}
        return example

from utils import *
leetcode_en_solutions = list(stream_jsonl("/home/ma-user/work/liurb/raw_leetcode/leetcode_en_solutions.jsonl"))

#多进程实现
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers = 16) as executor:
    futures = []
    for sample in tqdm(leetcode_en_solutions, total=len(leetcode_en_solutions), desc="Submitting tasks"):
        future = executor.submit(map_solution, sample)
        futures.append(future)
    
    results = []
    for future in tqdm(as_completed(futures), total=len(futures), desc="Completing tasks"):
        result = future.result()
        results.append(result)

write_jsonl("leetcode_solution_meta.jsonl", results)