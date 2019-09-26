import json
import os
import numpy
import pickle
import requests


def run_tasks(dir_path, task_num=None):
    tasks, filenames = [], [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    for filename in filenames:
        if filename.endswith(".json"):
            with open(filename) as f:
                data = json.load(f)
                resp = requests.post('http://localhost:8000/take_exam', json={'tasks':data})

                print(resp.json())
                #tasks += [d for d in data if 'id' in d and int(d['id']) == task_num]
    return #tasks


#tasks = load_tasks('public_set/train', task_num=8)




print(requests.get('http://localhost:8000/ready'))


#resp = requests.post('http://localhost:8000/take_exam', json={'tasks':tasks[:30]})

#print(resp.json())

run_tasks('public_set/test')
