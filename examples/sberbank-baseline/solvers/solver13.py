import random
import pickle
import numpy as np
from fuzzywuzzy import fuzz
from operator import itemgetter
from string import punctuation
from solvers.utils import BertEmbedder
from sklearn.metrics.pairwise import cosine_similarity


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.representatives = {}

    def init_seed(self):
        return random.seed(self.seed)

    def get_target(self, task):
        solution = task["solution"]["correct_variants"] if "correct_variants" in task["solution"] \
            else [task["solution"]["correct"]]
        return solution

    def process_task(self, task):
        sentences = [t.strip() for t in task["text"].lower().replace("…", ".").replace("!", ".").split(".")
                     if "(не)" in t]
        words = [[w.strip(",.") for w in s.split() if "(не)" in w][0] for s in sentences]
        return words

    def get_representatives(self, word, threshold=75):
        representatives = [rep for rep in self.representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_similarity(self, word, representatives):
        x = self.token_embedding([word]).reshape(1, -1)
        y = [self.representatives[rep] for rep in representatives]
        similarity = max([cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y])
        return similarity

    def fit(self, train):
        for task in train:
            solution = self.get_target(task)
            for word in solution:
                word = word.strip(punctuation).replace("не", "(не)")
                if word not in self.representatives:
                    self.representatives[word] = self.token_embedding([word])

    def save(self, path="data/models/solver13.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)

    def load(self, path="data/models/solver13.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        predictions, words = {}, self.process_task(task)
        for word in words:
            representatives = self.get_representatives(word)
            if representatives:
                similarity = self.get_similarity(word, representatives)
                word = word.replace("(не)", "не").strip(punctuation)
                predictions[word] = similarity
        return max(predictions.items(), key=itemgetter(1))[0] if predictions else words[0].replace("(не)", "не").strip(
            punctuation)