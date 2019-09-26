import random
import re
import time
import joblib
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from solvers.utils import BertEmbedder
from utils import read_config


class Solver(BertEmbedder):

    def __init__(self, seed=42, model_config="data/models/model_26.json"):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.model_config = model_config
        self.config = read_config(self.model_config)
        self.unified_substrings = self.config["unified_substrings"]
        self.replacements = self.config["replacements"]
        self.duplicates = self.config["duplicates"]
        self.classifier = LogisticRegression(verbose=10)
        self.label_encoder = LabelEncoder()

    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        decisions, phrases = dict(), self.extract_phrases(task)
        used_answers, choices = set(), [self.unify_type(choice["text"]) for choice in task["question"]["choices"]]
        for letter in "ABCD":
            if len(phrases[letter]) == 0:
                decisions[letter] = "1"
            else:
                embedding = np.mean(np.vstack(self.sentence_embedding(phrases[letter])), 0)
                proba = self.classifier.predict_proba(embedding.reshape((1, -1)))[0]
                options = list(self.label_encoder.inverse_transform(np.argsort(proba)))[::-1]
                try:
                    answer = next(option for option in options if option in choices and option not in used_answers)
                except StopIteration:
                    decisions[letter] = "1"
                    continue
                used_answers.add(answer)
                answer_id = str(choices.index(answer) + 1)
                decisions[letter] = answer_id
        return decisions

    def unify_type(self, type_):
        type_ = re.split(r"\s", type_, 1)[-1]
        type_ = type_.strip(" \t\n\v\f\r-–—−()").replace("и ", "")
        for key, value in self.unified_substrings.items():
            if key in type_:
                return value
        for key, value in self.replacements.items():
            type_ = re.sub(key + r"\b", value, type_)
        for duplicate_list in self.duplicates:
            if type_ in duplicate_list:
                return duplicate_list[0]
        return type_

    @staticmethod
    def get_sent_num(sent: str):
        match = re.search(r"\(([\dЗбOО]{1,2})\)", sent)
        if match:
            num = match.group(1)
            table = str.maketrans("ЗбОO", "3600")
            num = num.translate(table)
            num = int(num)
            return num
        match = re.search(r"([\dЗбOО]{1,2})\)", sent)
        if match:
            num = match.group(1)
            table = str.maketrans("ЗбОO", "3600")
            num = num.translate(table)
            num = int(num)
            return num

    def extract_phrases(self, task):
        result, text = {key: list() for key in "ABCD"}, task["text"]
        text = text.replace("\xa0", " ")
        citations = [sent for sent in sent_tokenize(text.split("Список терминов")[0])
                     if re.search(r"\([А-Г]\)|\(?[А-Г]\)?_{2,}", sent)]
        text = [x for x in re.split(r"[АA]БВГ\.?\s*", text) if x != ""][-1]
        text = re.sub(r"(\([\dЗбOО]{1,2}\))", r" \1 ", text)
        sents = sent_tokenize(text)
        sents = [x.strip() for sent in sents for x in re.split(r"…|\.\.\.", sent)]
        sents = [x.strip() for sent in sents for x in re.split(" (?=\([\dЗбОO])", sent)]
        sents = [sent for sent in sents if re.match(r"\s*\(?[\dЗбОO]{1,2}\)", sent)]
        assert all(re.search(r"\({}\)|\(?{}\)?_{2,}".replace("{}", letter), ' '.join(citations))
                   for letter in "АБВГ"), "Not all letters found in {}".format(citations)
        citations = " ".join(citations)
        citations = re.split("\([А-Г]\)|\(?[А-Г]\)?_{2,}", citations)[1:]
        assert len(citations) == 4, "Expected 4 (not {}) citations: {}".format(len(citations), citations)
        for citation, letter in zip(citations, "ABCD"):
            sent_nums = list()
            matches = re.finditer(r"предложени\w{,3}\s*(\d[\d\-— ,]*)", citation)
            for match in matches:
                sent_nums_str = match.group(1)
                for part in re.split(r",\s*", sent_nums_str):
                    part = part.strip(" \t\n\v\f\r-–—−")
                    if len(part) > 0:
                        if part.isdigit():
                            sent_nums.append(int(part))
                        else:
                            from_, to = re.split(r"[-–—−]", part)
                            extension = range(int(from_), int(to) + 1)
                            sent_nums.extend(extension)
            sents_ = [sent for sent in sents if self.get_sent_num(sent) in sent_nums]
            sents_ = [re.sub(r"(\([\dЗбOО]{1,2}\))\s*", "", sent) for sent in sents_]
            result[letter].extend(sents_)
            matches = re.finditer(r"[«\"](.*?)[»\"]", citation)
            for match in matches:
                result[letter].append(match.group(1))
        result = {key: list(set(value)) for key, value in result.items()}
        return result

    def fit(self, tasks):
        self.corpus, self.types = list(), list()
        for task in tasks:
            letters_to_phrases = self.extract_phrases(task)
            for key in "ABCD":
                questions = letters_to_phrases[key]
                answer_number = task["solution"]["correct"][key]
                answer = next(
                    answ["text"] for answ in task["question"]["choices"] if
                    answ["id"] == answer_number)
                if answer.isdigit():
                    continue
                answer = self.unify_type(answer)
                self.corpus.extend(questions)
                self.types.extend([answer] * len(questions))
        start = time.time()
        print("Encoding sentences with bert...")
        X = np.vstack(self.sentence_embedding(self.corpus))
        print("Encoding finished. This took {} seconds".format(time.time() - start))
        y = self.label_encoder.fit_transform(self.types)
        self.classifier.fit(X, y)

    def load(self, path="data/models/solver26.pkl"):
        model = joblib.load(path)
        self.classifier = model["classifier"]
        self.label_encoder = model["label_encoder"]

    def save(self, path="data/models/solver26.pkl"):
        model = {"classifier": self.classifier,
                 "label_encoder": self.label_encoder}
        joblib.dump(model, path)
