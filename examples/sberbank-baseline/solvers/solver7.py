import random
import pickle
import numpy as np
from string import punctuation
from fuzzywuzzy import fuzz, process
from solvers.utils import BertEmbedder
from sklearn.metrics.pairwise import cosine_similarity


class Solver(BertEmbedder):

    def __init__(self, seed=42, rnc_path="data/1grams-3.txt"):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.representatives = {}
        self.rnc_path = rnc_path
        self.rnc_unigrams = self.lazy_unigrams(self.rnc_path)

    def init_seed(self):
        return random.seed(self.seed)

    @staticmethod
    def lazy_unigrams(rnc_path):
        unigram_freq_dict = {}
        for line in open(rnc_path, "r", encoding="utf-8").read().split("\n"):
            pair = line.lower().split("\t")
            try:
                freq, unigram = int(pair[0]), " ".join([el for el in pair[1:]
                                                        if el is not "" and el not in punctuation])
                if unigram not in unigram_freq_dict:
                    unigram_freq_dict[unigram] = freq
                else:
                    unigram_freq_dict[unigram] += freq
            except ValueError:
                pass
        return unigram_freq_dict

    def get_target(self, task):
        solution = task["solution"]["correct_variants"] if "correct_variants" in task["solution"] else [
            task["solution"]["correct"]]
        return solution

    def process_task(self, task):
        text = " ".join([t for t in task["text"].split(".") if "\n" in t])
        words = [w.strip(punctuation).split() for w in text.split("\n") if "Исправьте" not in w and len(w) > 1]
        words = [" ".join([w.lower() for w in words_ if w.isupper()]) for words_ in words]
        return words

    def get_representatives(self, word, threshold=65):
        representatives = [rep for rep in self.representatives if fuzz.ratio(word, rep) >= threshold]
        return representatives

    def get_error_word(self, words):
        frequencies = [self.rnc_unigrams[w] if w in self.rnc_unigrams else 10 for w in words]
        error_word = words[np.argmin(frequencies)]
        return error_word

    def get_similarity(self, word, representatives):
        x, y = self.token_embedding([word]).reshape(1, -1), [self.representatives[rep] for rep in representatives]
        similarities = [cosine_similarity(x, y_.reshape(1, -1))[0][0] for y_ in y]
        prediction = representatives[np.argmax(similarities)]
        return prediction

    def fit(self, train):
        for task in train:
            words, solution = self.process_task(task), self.get_target(task)
            error_word = process.extractOne(solution[0], words)[0]
            for word in words:
                if word != error_word and word not in self.representatives:
                    self.representatives[word] = self.token_embedding([word])
            for correct in solution:
                if correct not in self.representatives:
                    self.representatives[correct] = self.token_embedding([correct])

    def save(self, path="data/models/solver7.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.representatives, f)
    
    def load(self, path="data/models/solver7.pkl"):
        with open(path, "rb") as f:
            self.representatives = pickle.load(f)

    def predict_from_model(self, task):
        words = self.process_task(task)
        error_word = self.get_error_word(words)
        representatives = self.get_representatives(error_word)
        if representatives:
            prediction = self.get_similarity(error_word, representatives)
            return prediction
        return words[0].strip(punctuation)