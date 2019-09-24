import re
import random
from fuzzywuzzy import fuzz
import pickle
from string import punctuation
from operator import itemgetter
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
        text = re.sub(r"([а-яё]+)\.([а-яё]+)", r"\1. \2", task["text"].lower().replace(".(", ". (").replace("?", ". "))
        words = [w.replace("(", "").replace(")", "") for w in
                 [w.strip(punctuation) for w in text.split() if any([s in w for s in ("(", ")")])]]
        return words

    def get_representatives(self, word, threshold=75):
        representatives = [rep for rep in self.representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_similarity(self, word, representatives):
        x, y = self.token_embedding([word]).reshape(1, -1), [self.representatives[rep] for rep in representatives]
        similarity = max([cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y])
        return similarity

    def fit(self, train):
        for task in train:
            words, solution = self.process_task(task), self.get_target(task)
            if len(words) == 10:
                for i in range(0, len(words), 2):
                    word1, word2 = words[i], words[i+1]
                    candidate = word1 + word2
                    if candidate in solution:
                        if word1 not in self.representatives:
                            self.representatives[word1] = self.token_embedding([word1])
                        if word2 not in self.representatives:
                            self.representatives[word2] = self.token_embedding([word2])

    def save(self, path="data/models/solver14.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)
    
    def load(self, path="data/models/solver14.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        predictions = {}
        words = self.process_task(task)
        if len(words) == 10:
            for i in range(0, len(words), 2):
                word1, word2 = words[i], words[i + 1]
                rep1, rep2 = self.get_representatives(word1), self.get_representatives(word2)
                if rep1 and rep2:
                    cos1 = self.get_similarity(word1, rep1)
                    cos2 = self.get_similarity(word2, rep2)
                    predictions[word1 + word2] = cos1 + cos2
        return max(predictions.items(), key=itemgetter(1))[0] if predictions else words[0]