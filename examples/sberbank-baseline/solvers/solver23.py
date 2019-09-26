from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import *
from sklearn import pipeline
import pymorphy2
import random
import joblib
import re
import os


class Solver(object):
    def __init__(self, seed=42):
        self.seed = seed
        self.init_seed()
        self.has_model = True
        self.possible_mark_choices = set()
        self.russian_stopwords = stopwords.words("russian")
        self.morph = pymorphy2.MorphAnalyzer()
        self.cls_dict = {
            "description": {"classifier": "clf_desc", "train": "description"},
            "narrative": {"classifier": "clf_nar", "train": "narrative"},
            "discourse": {"classifier": "clf_discource", "train": "discourse"},
            "cause": {"classifier": "clf_cause", "train": "cause"},
            "general": {"classifier": "clf_gen", "train": "general_class"}
        }
        
    def load(self, path='data/models'):
        if path!='data/models':
            path = os.path.dirname(path)
        self.clf_gen = joblib.load( os.path.join(path, 'clf_gen.joblib'))
        self.clf_nar = joblib.load( os.path.join(path, 'clf_nar.joblib'))
        self.clf_desc = joblib.load( os.path.join(path, 'clf_desc.joblib'))
        self.clf_discource = joblib.load( os.path.join(path, 'clf_discource.joblib'))
        self.clf_cause = joblib.load( os.path.join(path, 'clf_cause.joblib'))


    def save(self, path='data/models'):
        if path!='data/models':
            path = os.path.dirname(path)
        joblib.dump(self.clf_gen, os.path.join(path, 'clf_gen.joblib'))
        joblib.dump(self.clf_nar, os.path.join(path, 'clf_nar.joblib'))
        joblib.dump(self.clf_desc, os.path.join(path, 'clf_desc.joblib'))
        joblib.dump(self.clf_discource, os.path.join(path, 'clf_discource.joblib'))
        joblib.dump(self.clf_cause, os.path.join(path, 'clf_cause.joblib'))

    def init_seed(self):
        return random.seed(self.seed)

    def parse_task(self, task):
        assert task["question"]["type"] == "multiple_choice"

        choices = task["question"]["choices"]
        description = task["text"]
        if "(1)" in description:
            text = "(1)" + " ".join(description.split("(1)")[1:])
        else:
            text = "(1)" + " ".join(description.split("1)")[1:])

        for choice in choices:
            self.possible_mark_choices.add(choice["text"])

        return text, choices

    def get_sentence_from_text(self, text, sent_num):
        if sent_num is not None:
            num = "({})".format(sent_num)
            try:
                sentence_text = text.split(num)[1]
            except IndexError:
                return ""
            match = re.search("[0-9]+\\)", sentence_text)
            if match is None:
                return ""
            else:
                split_by = sentence_text.split(match.group(0))[1]
                sentence = sentence_text.split(split_by)[0]
                return sentence[:-4]

    def get_sent_range_from_text(self, text, sent_num1, sent_num2):
        text = text.replace("(1 Т", "(11) Т")
        try:
            num1 = "({})".format(sent_num1)
            num2 = "({})".format(int(sent_num2) + 1)
        except ValueError:
            return ""
        try:
            sentence_text = text.split(num1)[1]
            sentence = sentence_text.split(num2)[0]
            return sentence
        except IndexError:
            return ""

    def assign_class(self, choice):
        if "рассужден" in choice:
            label = 'discourse'
        elif "повествова" in choice:
            label = 'narrative'
        elif "описа" in choice:
            label = "description"
        elif "следствие" in choice or "причин" in choice:
            label = "cause"
        elif "противопоставлен" in choice:
            label = "against"
        elif "ответ" in choice:
            label = "answer"
        elif "действ" in choice or "событ" in choice:
            label = "action"
        else:
            label = "general"
        return label

    def parse_one(self, text, sent_match):
        if "-" in sent_match[0]:
            id1, id2 = sent_match[0].split("-")
            sents = self.get_sent_range_from_text(text, id1, id2)
        else:
            sents = self.get_sentence_from_text(text, sent_match[0])
        return sents

    def parse_ranges(self, text, sent_match):
        if "-" in sent_match[0] and "-" in sent_match[1]:
            id1, id2 = sent_match[0].split("-")
            sent1 = self.get_sent_range_from_text(text, id1, id2)
            id1, id2 = sent_match[1].split("-")
            sent2 = self.get_sent_range_from_text(text, id1, id2)
        elif "-" in sent_match[0]:
            id1, id2 = sent_match[0].split("-")
            sent1 = self.get_sent_range_from_text(text, id1, id2)
            sent2 = self.get_sentence_from_text(text, sent_match[-1])
        elif "-" in sent_match[1]:
            id1, id2 = sent_match[1].split("-")
            sent1 = self.get_sent_range_from_text(text, id1, id2)
            sent2 = self.get_sentence_from_text(text, sent_match[0])
        else:
            sent1 = self.get_sentence_from_text(text, sent_match[0])
            sent2 = self.get_sentence_from_text(text, sent_match[-1])
        sents = [sent1, sent2]
        return sents

    def parse_choices(self, text, choice):
        choice = (
            re.sub("^\s?[0-9]+\\)\s?", "", choice).replace(", ", " ").replace(".", "")
        )
        choice = choice.replace('–', '-')
        sent_match = re.findall("[0-9\\-]+", choice)
        label = self.assign_class(choice)

        if len(sent_match) > 1:
            sent_match = [s for s in sent_match if s != "-"]
            if len(sent_match) == 1:
                id1, id2 = sent_match[0].split("-")
                return self.get_sent_range_from_text(text, id1, id2), label
            else:
                return self.parse_ranges(text, sent_match), label

        elif len(sent_match) == 0:
            return text, label
        else:
            sents = self.parse_one(text, sent_match)
            return sents, label

    def ensemble(self, X, y, label):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, stratify=y,
                                                                                random_state=7)

        func = self.cls_dict[label]["train"]
        predictions = getattr(self, func)()
        """
        print("Precision: {0:6.2f}".format(precision_score(self.y_test, predictions, average='macro')))
        print("Recall: {0:6.2f}".format(recall_score(self.y_test, predictions, average='macro')))
        print("F1-measure: {0:6.2f}".format(f1_score(self.y_test, predictions, average='macro')))
        print("Accuracy: {0:6.2f}".format(accuracy_score(self.y_test, predictions)))
        print(classification_report(self.y_test, predictions))
        """

    def preprocess(self, text):
        result = []
        text = re.sub("\\([0-9]+\\)", "", text)
        for word in text.split():
            word = word.strip(".,!)(*;:»«")
            if word not in self.russian_stopwords:
                anal = self.morph.parse(word)[0]
                lemma = anal.normal_form
                pos = anal.tag.POS
                if pos:
                    result.append(lemma + "_" + pos)
        return result

    def fit(self, tasks):
        self.load()
        return
        for_train_dict = defaultdict(lambda: defaultdict(list))
        for task in tasks:
            text, choices = self.parse_task(task)
            y_true = (
                task["solution"]["correct_variants"]
                if "correct_variants" in task["solution"]
                else [task["solution"]["correct"]]
            )

            for choice in choices:
                target = 1 if choice["id"] in y_true[0] else 0
                sents, label = self.parse_choices(text, choice["text"])
                if sents:
                    if isinstance(sents, list):
                        if None not in sents:
                            sents = "\n".join(sents)
                            if target == 1:
                                for_train_dict[label]["1"].append(sents)
                            else:
                                for_train_dict[label]["0"].append(sents)
                    else:
                        if target == 1:
                            for_train_dict[label]["1"].append(sents)
                        else:
                            for_train_dict[label]["0"].append(sents)


        for key, value in for_train_dict.items():
            if key in ["discourse", "description", "narrative", "cause", "general"]:
                X, y = [], []

                for k, val in value.items():
                    for v in val:
                        X.append(" ".join(self.preprocess(v)))
                        y.append(int(k))
                self.ensemble(X, y, key)



    def get_prediction(self, sents, label):
        if label == "action":
            n = 0
            for sent in sents:
                grammar = self.preprocess(sent)
                for lem_pos in grammar:
                    if "VERB" in lem_pos:
                        n += 1
            if n > 2:
                return True
        elif label == "answer":
            for sent in sents:
                if "?" in sent:
                    return True
        elif "against" == label:
            for sent in sents:
                if "но " in sent.lower():
                    return True
        else:
            s = [sent for sent in sents if sent is not None]
            X_test = [" ".join(self.preprocess(" ".join(s)))]

            func = self.cls_dict[label]["classifier"]
            predictions = getattr(self, func).predict(X_test)
            if predictions[0] == 1:
                return True

        return False

    def predict_from_model(self, task):
        """ Mean accuracy: 54% """
        text, choices = self.parse_task(task)

        result = []
        for choice in choices:
            sents, label = self.parse_choices(text, choice["text"])
            if isinstance(sents, str):
                sents = [sents]
            var = self.get_prediction(sents, label)
            if var:
                result.append(choice["id"])
        return result

    def predict_random(self, task):
        choices = task["question"]["choices"]
        pred = []
        for _ in range(random.choice([2, 3])):
            choice = random.choice(choices)
            pred.append(choice["id"])
            choices.remove(choice)

        return pred

    def predict(self, task):
        if not self.has_model:
            return self.predict_random(task)
        else:
            return self.predict_from_model(task)

    def description(self):
        self.clf_desc = pipeline.Pipeline([
            ('vect', CountVectorizer(max_features=1000, ngram_range=(1, 5), analyzer='word', lowercase=True)),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', linear_model.LogisticRegression(solver='lbfgs', random_state=1))
        ])

        self.clf_desc.fit(self.X_train, self.y_train)
        predictions = self.clf_desc.predict(self.X_test)
        return predictions

    def narrative(self):
        self.clf_nar = pipeline.Pipeline([
            ('vect', CountVectorizer(max_features=1000, ngram_range=(1, 5), analyzer='word', lowercase=True)),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', linear_model.LogisticRegression(solver='lbfgs', random_state=1))
        ])

        self.clf_nar.fit(self.X_train, self.y_train)
        predictions = self.clf_nar.predict(self.X_test)
        return predictions

    def discourse(self):
        self.clf_discource = pipeline.Pipeline([
            ('vect', CountVectorizer(max_features=1000, ngram_range=(1, 5), analyzer='word', lowercase=True)),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', linear_model.LogisticRegression(solver='lbfgs', random_state=1))
        ])

        self.clf_discource.fit(self.X_train, self.y_train)
        predictions = self.clf_discource.predict(self.X_test)
        return predictions

    def cause(self):
        self.clf_cause = pipeline.Pipeline([
            ('vect', CountVectorizer(max_features=1000, ngram_range=(1, 5), analyzer='word', lowercase=True)),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', linear_model.LogisticRegression(solver='lbfgs', random_state=1))
        ])

        self.clf_cause.fit(self.X_train, self.y_train)
        predictions = self.clf_cause.predict(self.X_test)
        return predictions

    def general_class(self):
        self.clf_gen = pipeline.Pipeline([
            ('vect', CountVectorizer(max_features=1000, ngram_range=(1, 5), analyzer='word', lowercase=True)),
            ('tfidf', TfidfTransformer(sublinear_tf=True)),
            ('clf', linear_model.LogisticRegression(solver='lbfgs', random_state=1))
        ])

        self.clf_gen.fit(self.X_train, self.y_train)
        predictions = self.clf_gen.predict(self.X_test)
        return predictions