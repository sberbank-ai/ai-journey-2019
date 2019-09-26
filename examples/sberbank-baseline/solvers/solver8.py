from ufal.udpipe import Model, Pipeline
from difflib import SequenceMatcher
from string import punctuation
import pymorphy2
import random
import re
import sys


def get_gerund(features):
    """деепричастие """
    hypothesys = []

    for feature in features:
        for row in feature:
            if row[4] == "VERB":
                if "VerbForm=Conv" in row[5]:
                    hypothesys.append(" ".join([row[2] for row in feature]))

    return hypothesys

def get_indirect_speech(features):
    """ косвенная речь """
    hypothesys = []
    for feature in features:
        for row in feature:
            if row[8] == '1':
                hypothesys.append(" ".join([row[2] for row in feature]))
    return hypothesys

def get_app(features):
    """ Приложение """
    hypothesys = []
    for feature in features:
        for row1, row2, row3 in zip(feature, feature[1:], feature[2:]):
            if row1[2] == "«" and row3[2] == "»" and row2[1] == '1':
                hypothesys.append(" ".join([row[2] for row in feature]))
            if "«" in row1[2]:
                if row1[2][1:][0].isupper():
                    hypothesys.append(" ".join([row[2] for row in feature]))
    return hypothesys

def get_predicates(features):
    """ связь подлежащее сказуемое root + subj = number """
    hypothesys = set()

    for feature in features:
        head, number = None, None
        for row in feature:
            if row[7] == 'root':
                head = row[0]
                for s in row[5].split('|'):
                    if "Number" in s:
                        number = s.replace("Number=", "")
        for row in feature:
            row_number = None
            for s in row[5].split('|'):
                if "Number" in s:
                    row_number = s.replace("Number=", "")
            if row[0] == head and number != row_number:
                hypothesys.add(" ".join([row[2] for row in feature]))
    return hypothesys

def get_clause(features):
    """ сложные предложения """
    hypothesys = set()
    for feature in features:
        for row in feature:
            if row[3] == 'который':
                hypothesys.add(" ".join([row[2] for row in feature]))
    return hypothesys


def get_participle(features):
    """причастие """
    hypothesys = []
    for feature in features:
        for row in feature:
            if row[4] == "VERB":
                if "VerbForm=Part" in row[5]:
                    hypothesys.append(" ".join([row[2] for row in feature]))
    return hypothesys

def get_verbs(features):
    """ вид и время глаголов """
    hypothesys = set()
    for feature in features:
        head, aspect, tense = None, None, None
        for row in feature:
            if row[7] == 'root':
                # head = row[0]
                for s in row[5].split('|'):
                    if "Aspect" in s:
                        aspect = s.replace("Aspect=", "")
                    if "Tense" in s:
                        tense = s.replace("Tense=", "")

        for row in feature:
            row_aspect, row_tense = None, None
            for s in row[5].split('|'):
                if "Aspect" in s:
                    row_aspect = s.replace("Aspect=", "")
            for s in row[5].split('|'):
                if "Tense" in s:
                    row_tense = s.replace("Tense=", "")
            if row[4] == "VERB" and row_aspect != aspect: # head ?
                hypothesys.add(" ".join([row[2] for row in feature]))

            if row[4] == "VERB" and row_tense != tense:
                hypothesys.add(" ".join([row[2] for row in feature]))
    return hypothesys

def get_nouns(features):
    """ формы существительных ADP + NOUN"""
    hypothesys = set()
    apds = ["благодаря", "согласно", "вопреки", "подобно", "наперекор",
            "наперерез", "ввиду", "вместе", "наряду", "по"]
    for feature in features:
        for row1, row2 in zip(feature, feature[1:]):
            if row1[3] in apds:
                if row2[4] == 'NOUN':
                    hypothesys.add(" ".join([row[2] for row in feature]))
    return hypothesys

def get_numerals(features):
    hypothesys = []
    for feature in features:
            for row in feature:
                if row[4] == "NUM":
                    hypothesys.append(" ".join([row[2] for row in feature]))
    return hypothesys


def get_homogeneous(features):
    hypothesys = set()
    for feature in features:
        sent = " ".join([token[2] for token in feature]).lower()
        for double_conj in ["если не", "не столько", "не то чтобы"]:
            if double_conj in sent:
                hypothesys.add(sent)
    return hypothesys


class Solver():

    def __init__(self, seed=42):
        self.morph = pymorphy2.MorphAnalyzer()
        self.categories = set()
        self.has_model = True
        self.model = Model.load("data/udpipe_syntagrus.model".encode())
        self.process_pipeline = Pipeline(self.model, 'tokenize'.encode(), Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu'.encode())
        self.seed = seed
        self.label_dict = {
            'деепричастный оборот': "get_gerund",
            'косвенный речь': "get_indirect_speech",
            'несогласованный приложение': "get_app",
            'однородный член': "get_homogeneous",
            'причастный оборот': "get_participle",
            'связь подлежащее сказуемое': "get_predicates",
            'сложноподчинённый': "get_clause",
            'сложный': "get_clause",
            'соотнесённость глагольный форма': "get_verbs",
            'форма существительное': "get_nouns",
            'числительное': "get_numerals"
        }
        self.init_seed()

    def init_seed(self):
        return random.seed(self.seed)

    def get_syntax(self, text):
        processed = self.process_pipeline.process(text.encode())
        content = [l for l in processed.decode().split('\n') if not l.startswith('#')]
        tagged = [w.split('\t') for w in content if w]
        return tagged

    def tokens_features(self, some_sent):

        tagged = self.get_syntax(some_sent)
        features = []
        for token in tagged:
            _id, token, lemma, pos, _, grammar, head, synt, _, _, = token #tagged[n]
            capital, say = "0", "0"
            if lemma[0].isupper():
                    capital = "1"
            if lemma in ["сказать", "рассказать", "спросить", "говорить"]:
                    say = "1"
            feature_string = [_id, capital, token, lemma, pos, grammar, head, synt, say]
            features.append(feature_string)
        return features

    def normalize_category(self, cond):
        """ {'id': 'A', 'text': 'ошибка в построении сложного предложения'} """
        condition = cond["text"].lower().strip(punctuation)
        condition = re.sub("[a-дabв]\)\s", "", condition).replace('членами.', "член")
        norm_cat = ""
        for token in condition.split():
            lemma = self.morph.parse(token)[0].normal_form
            if lemma not in [
                    "неправильный", "построение", "предложение", "с", "ошибка", "имя",
                    "видовременной", "видо-временной", "предложно-падежный", "падежный",
                    "неверный", "выбор", "между", "нарушение", "в", "и", "употребление",
                    "предлог", "видовременный", "временной"
                ]:
                norm_cat += lemma + ' '
        self.categories.add(norm_cat[:-1])
        return norm_cat

    def parse_task(self, task):

        assert task["question"]["type"] == "matching"

        conditions = task["question"]["left"]
        choices = task["question"]["choices"]

        good_conditions = []
        X = []
        for cond in conditions:  # LEFT
            good_conditions.append(self.normalize_category(cond))
                    
        for choice in choices:
            choice = re.sub("[0-9]\\s?\)", "", choice["text"])
            X.append(choice)
        return X, choices, good_conditions

    def match_choices(self, label2hypothesys, choices):
        final_pred_dict = {}
        for key, value in label2hypothesys.items():
            if len(value) == 1:
                variant = list(value)[0]
                variant = variant.replace(' ,', ',')
                for choice in choices:
                    ratio = SequenceMatcher(None, variant, choice["text"]).ratio()
                    if ratio > 0.9:
                        final_pred_dict[key] = choice["id"]
                        choices.remove(choice)

        for key, value in label2hypothesys.items():
            if key not in final_pred_dict.keys():
                variant = []
                for var in value:
                    for choice in choices:
                        ratio = SequenceMatcher(None, var, choice["text"]).ratio()
                        if ratio > 0.9:
                            variant.append(var)
                if variant:
                    for choice in choices:
                        ratio = SequenceMatcher(None, variant[0], choice["text"]).ratio()
                        if ratio > 0.9:
                            final_pred_dict[key] = choice["id"]
                            choices.remove(choice)
                else:
                    variant = [choice for choice in choices]
                    if variant:
                        final_pred_dict[key] = variant[0]["id"]
                        for choice in choices:
                            ratio = SequenceMatcher(None, variant[0]["text"], choice["text"]).ratio()
                            if ratio > 0.9:
                                choices.remove(choice)

        for key, value in label2hypothesys.items():
            if key not in final_pred_dict.keys():
                variant = [choice for choice in choices]
                if variant:
                    final_pred_dict[key] = variant[0]["id"]
                    for choice in choices:
                        ratio = SequenceMatcher(None, variant[0]["text"], choice["text"]).ratio()
                        if ratio > 0.9:
                            choices.remove(choice)
        return final_pred_dict

    def predict_random(self, task):
        """ Test a random choice model """
        conditions = task["question"]["left"]
        choices = task["question"]["choices"]
        pred = {}
        for cond in conditions:
            pred[cond["id"]] = random.choice(choices)["id"]
        return pred

    def predict(self, task):
        if not self.has_model:
            return self.predict_random(task)
        else:
            return self.predict_from_model(task)

    def fit(self, tasks):
        pass

    def load(self, path="data/models/solver8.pkl"):
        pass

    def save(self, path="data/models/solver8.pkl"):
        pass

    def predict_from_model(self, task):
        x, choices, conditions = self.parse_task(task)
        all_features = []
        for row in x:
            all_features.append(self.tokens_features(row))

        label2hypothesys = {}
        for label in self.label_dict.keys():
            func = self.label_dict[label.rstrip()]
            hypotesis = getattr(sys.modules[__name__], func)(all_features)
            label2hypothesys[label] = hypotesis

        final_pred_dict = self.match_choices(label2hypothesys, choices)
        
        pred_dict = {}
        for cond, key in zip(conditions, ["A", "B", "C", "D", "E"]):
            cond = cond.rstrip()
            try:
                pred_dict[key] = final_pred_dict[cond]
            except KeyError:
                pred_dict[key] = "1"
        return pred_dict