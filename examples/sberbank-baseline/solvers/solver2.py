import random
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
from sklearn.exceptions import NotFittedError
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from solvers.utils import BertEmbedder
from string import punctuation
import joblib


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.is_train_task = False
        self.seed = seed
        self.init_seed()
        self.classifier = MLPClassifier(max_iter=300)
        self.label_encoder = LabelEncoder()
        self.default_word = None
        self.fitted = False

    def init_seed(self):
        random.seed(self.seed)
        
    def save(self, path="data/models/solver2.pkl"):
        model = {"classifier": self.classifier,
                 "label_encoder": self.label_encoder,
                 "default_word": self.default_word}
        joblib.dump(model, path)

    def load(self, path="data/models/solver2.pkl"):
        model = joblib.load(path)
        self.classifier = model["classifier"]
        self.label_encoder = model["label_encoder"]
        self.default_word = model["default_word"]
        self.fitted = True

    @staticmethod
    def get_close_sentence(text):
        sentences = sent_tokenize(text)
        if any("<...>" in sent or "<…>" in sent for sent in sentences):
            num = next(num for num, sent in enumerate(sentences) if "<...>" in sent or "<…>" in sent)
            return ' '.join(sentences[num-1:num+1])
        else:
            try:
                num = next(num for num, sent in enumerate(sentences) if ("..." in sent or "…" in sent)
                           and not sent.endswith("...") and not sent.endswith("…"))
                return ' '.join(sentences[num - 1:num + 1])
            except StopIteration:
                return None

    def fit(self, tasks):
        X, y = list(), list()
        for task in tasks:
            text = task.get("text")
            if text is None:
                continue
            close = self.get_close_sentence(text)
            if close is None:
                continue
            correct = task["solution"]["correct_variants"] if "correct_variants" in task["solution"] else [
                task["solution"]["correct"]]
            for variant in correct:
                X.append(close)
                y.append(variant)
        self.default_word = Counter(y).most_common(1)[0][0]
        X = np.vstack(self.sentence_embedding(X))
        y = self.label_encoder.fit_transform(y)
        self.classifier.fit(X, y)
        self.fitted = True

    def predict_from_model(self, task):
        if not self.fitted:
            raise NotFittedError
        if task.get("text") is None:
            return self.default_word
        close = self.get_close_sentence(task["text"])
        if close is None:
            return self.default_word
        X = np.vstack(self.sentence_embedding([task["text"]]))
        result = self.classifier.predict(X)[0]
        result = str(list(self.label_encoder.inverse_transform([result]))[0])
        return result.strip(punctuation)
