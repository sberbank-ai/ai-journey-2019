import random
import re
import nltk
import pymorphy2
from nltk.util import ngrams
from sklearn.metrics.pairwise import cosine_similarity
from solvers.utils import BertEmbedder
from string import punctuation


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.morph = pymorphy2.MorphAnalyzer()
        self.has_model = True
        self.mode = 1 # 1 - find wrong word, 2 - replace word

    def init_seed(self):
        return random.seed(self.seed)

    def predict_random(self, task_desc):
        """Random variant"""
        task_desc = re.sub("[^а-я0-9\-]", " ", task_desc)
        result = random.choice(task_desc.split())
        return result

    def exclude_word(self, task_sent):
        """Make it with Bert"""
        tokens = [token.strip('.,";!:?><)«»') for token in task_sent.split(" ") if token != ""]

        to_tokens = []
        for token in tokens:
            parse_res = self.morph.parse(token)[0]
            if parse_res.tag.POS not in ["CONJ", "PREP", "PRCL", "INTJ", "PRED", "NPRO"]:
                if parse_res.normal_form != 'быть':
                    to_tokens.append((parse_res.word, parse_res.tag.POS))

        bigrams = list(ngrams(to_tokens, 2))

        results = []
        for bigram in bigrams:
            if bigram[0] != bigram[1]:
                b1 = self.sentence_embedding([bigram[0][0]])[0].reshape(1, -1)
                b2 = self.sentence_embedding([bigram[1][0]])[0].reshape(1, -1)
                sim = cosine_similarity(b1, b2)[0][0]
                results.append((sim, bigram[0][0], bigram[1][0], bigram[0][1], bigram[0][1]))
        results = sorted(results)
        final_pair = results[-1]
        if final_pair[-1] == 'NOUN' and final_pair[-2] == 'NOUN':
            return results[-1][2], tokens
        else:
            return results[-1][1], tokens

    def fit(self, tasks):
        pass
        
    def load(self, path="data/models/solver6.pkl"):
        pass

    def save(self, path="data/models/solver6.pkl"):
        pass

    def predict(self, task):
        if not self.has_model:
            return self.predict_random(task)
        else:
            return self.predict_from_model(task)

    def predict_from_model(self, task):
        description = task["text"]
        task_desc = ""
        if "заменив" in description:
            self.mode = 2
        else:
            self.mode = 1
        for par in description.split("\n"):
            for sentence in nltk.sent_tokenize(par):
                sentence = sentence.lower().rstrip(punctuation).replace('6.', "")
                if re.match('.*(отредактируйте|выпишите|запишите|исправьте|исключите).*', sentence):
                    continue
                else:
                    task_desc += sentence
        result, tokens = self.exclude_word(task_desc)
        return result.strip(punctuation)
