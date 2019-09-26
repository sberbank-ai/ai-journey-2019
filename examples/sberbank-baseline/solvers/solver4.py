import re
import os
import random
from string import punctuation


class Solver(object):

    def __init__(self, seed=42, data_path='data/'):
        self.is_train_task = False
        self.seed = seed
        self.init_seed()
        self.stress = open(os.path.join(data_path, 'agi_stress.txt'), 'r', encoding='utf8').read().split('\n')[:-1]

    def init_seed(self):
        random.seed(self.seed)

    def predict(self, task):
        return self.predict_from_model(task)

    def compare_text_with_variants(self, variants, task_type='incorrect'):
        result = ''
        if task_type == 'incorrect':
            for variant in variants:
                if variant not in self.stress:
                    result = variant
        else:
            for variant in variants:
                if variant in self.stress:
                    result = variant
        if not variants:
            return ''
        if not result:
            result = random.choice(variants)
        return result.lower().strip(punctuation)

    def process_task(self, task):
        task_text = re.split(r'\n', task['text'])
        variants = task_text[1:-1]
        if 'Выпишите' in task_text[-1]:
            task = task_text[0] + task_text[-1]
        else:
            task = task_text[0]
        if 'неверно' in task.lower():
            task_type = 'incorrect'
        else:
            task_type = 'correct'
        return task_type, task, variants

    def fit(self, tasks):
        pass

    def load(self, path="data/models/solver4.pkl"):
        pass

    def save(self, path="data/models/solver4.pkl"):
        pass

    def predict_from_model(self, task):
        task_type, task, variants = self.process_task(task)
        result = self.compare_text_with_variants(variants, task_type)
        return result.strip()
