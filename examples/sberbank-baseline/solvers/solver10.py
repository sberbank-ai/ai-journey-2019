import re
import random
import pymorphy2
from solvers.utils import standardize_task, AbstractSolver


class Solver(object):
    """
    Solver for tasks 10, 11, 12
    """
    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.morph = pymorphy2.MorphAnalyzer()

    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        result, task = [], standardize_task(task)
        match = re.search(r'буква ([ЭОУАЫЕЁЮЯИ])', task["text"])
        if match:
            letter = match.group(1)
            return self.get_answer_by_vowel(task["question"]["choices"], letter.lower())
        elif "одна и та же буква" in task["text"]:
            for vowel in "эоуаыеёюяидтсз":
                result_with_this_vowel = self.get_answer_by_vowel(task["question"]["choices"], vowel)
                result.extend(result_with_this_vowel)
        return sorted(list(set(result)))

    def get_answer_by_vowel(self, choices, vowel):
        result = list()
        for choice in choices:
            parts = [re.sub(r"^\d\) ?| ?\(.*?\) ?", "", x) for x in choice["parts"]]
            parts = [x.replace("..", vowel) for x in parts]
            if all(self.morph.word_is_known(word) for word in parts):
                result.append(choice["id"])
        return sorted(result)

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass
