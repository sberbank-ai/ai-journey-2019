import json
import os
import numpy
import pickle
import pymorphy2
import re

def rus_tok(text, m = pymorphy2.MorphAnalyzer()):
    reg = '([0-9]|\W|[a-zA-Z])'
    toks = text.split()
    return [m.parse(t)[0].normal_form for t in toks if not re.match(reg, t)]

def load_tasks(dir_path, task_num=None):
    tasks, filenames = [], [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    for filename in filenames:
        if filename.endswith(".json"):
            with open(filename, encoding='utf-8') as f:
                dt = f.read().encode('utf-8')
                data = json.loads(dt)
                tasks += [d for d in data if 'id' in d and int(d['id']) == task_num]
    return tasks


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def random_choice(responses, probs=None):
    idx = numpy.random.choice(list(range(len(responses))), p=probs)
    return responses[idx]


def read_config(config):
    if isinstance(config, str):
        with open(config, "r", encoding="utf-8") as f:
            config = json.load(f)
    return config


def if_none(origin, other):
    return other if origin is None else origin


def singleton(cls):
    instance = None

    @wraps(cls)
    def inner(*args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = cls(*args, **kwargs)
        return instance
    return inner


def get_task_by_id(task_num, dir_path, print_errors=False, with_targets=True):
    res = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".json"):
            data = read_config(os.path.join(dir_path, file_name))
            task = list(filter(lambda x: str(x.get("id", -1)) == str(task_num) and
                                         (not with_targets or len(x['question'].get('choices',"")) >= 1), data))
            if len(task):
                task = task[0]
                task["file_path"] = file_name
                res.append(task)
            elif print_errors:
                print("Task {} doesn't exist at {}".format(task_num, file_name))
    return res
