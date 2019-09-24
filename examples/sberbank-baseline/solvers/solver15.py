import re
import random
import pickle
from fuzzywuzzy import fuzz
from string import punctuation
from operator import itemgetter
from solvers.utils import BertEmbedder
from sklearn.metrics.pairwise import cosine_similarity


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.representatives = {"n": {}, "nn": {}}

    def init_seed(self):
        return random.seed(self.seed)

    def get_target(self, task):
        solution = task["solution"]["correct_variants"][0] if "correct_variants" in task["solution"] else \
            task["solution"]["correct"]
        return solution

    def process_task(self, task):
        text = task["text"].replace("?", ".").replace("\xa0", "")
        placeholders = [choice["placeholder"] for choice in task["question"]["choices"]]
        words = [re.sub("[^а-яё]+", "0", word.strip(punctuation)) for word in text.split() if ")" in word
                 and any([n in word for n in placeholders])]
        n_number = text.split(".")[0].split()[-1].lower()
        return text, words, n_number

    def get_representatives(self, word, representatives, threshold=70):
        representatives = [rep for rep in representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_similarity(self, x, representatives):
        y = [self.representatives["n"][rep] if rep in self.representatives["n"]
             else self.representatives["nn"][rep] for rep in representatives]
        similarity = max([cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y])
        return similarity
    
    def parse_representatives(self, task):
        text, words, n_number = self.process_task(task)
        solution = self.get_target(task)
        if len(n_number) == 1:
            n_words = [re.sub("[^а-яё]+", "н", word.strip(punctuation)) for word in text.split()
                       if any([d in word for d in solution]) and ")" in word]
            for word in n_words:
                if word not in self.representatives["n"]:
                    self.representatives["n"][word] = self.token_embedding([word])
            for word in words:
                n_replacement = word.replace("0", "н")
                nn_replacement = word.replace("0", "нн")
                if n_replacement not in n_words and nn_replacement not in self.representatives["nn"]:
                    self.representatives["nn"][nn_replacement] = self.token_embedding([nn_replacement])
        elif len(n_number) == 2:
            nn_words = [re.sub("[^а-яё]+", "нн", word.strip(punctuation)) for word in text.split()
                        if any([d in word for d in solution]) and ")" in word]
            for word in nn_words:
                if word not in self.representatives["nn"]:
                    self.representatives["nn"][word] = self.token_embedding([word]) 
            for word in words:
                n_replacement = word.replace("0", "н")
                nn_replacement = word.replace("0", "нн")
                if nn_replacement not in nn_words and n_replacement not in self.representatives["n"]:
                    self.representatives["n"][n_replacement] = self.token_embedding([n_replacement])

    def fit(self, tasks):
        for task in tasks:
            self.parse_representatives(task)
            
    def save(self, path="data/models/solver15.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)
    
    def load(self, path="data/models/solver15.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        prediction = []
        text, words, n_number = self.process_task(task)
        for i, word in enumerate([word.replace("0", "н") for word in words]):
            representatives = {}
            x = self.token_embedding([word]).reshape(1, -1)
            c1 = self.get_representatives(word, self.representatives["n"])
            c2 = self.get_representatives(word, self.representatives["nn"])
            if c1:
                representatives["н"] = self.get_similarity(x, c1)
            if c2:
                representatives["нн"] = self.get_similarity(x, c2)
            if representatives:
                answer = max(representatives.items(), key=itemgetter(1))[0]
                if answer == n_number:
                    prediction.append(str(i + 1))
        return sorted(prediction) if prediction else ["1"]