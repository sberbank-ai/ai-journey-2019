import re
import pymorphy2
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from solvers.utils import AbstractSolver
import joblib


class Solver(AbstractSolver):
    def __init__(self, seed=42, train_size=0.85):
        self.has_model = False
        self.is_train_task = False
        self.morph = pymorphy2.MorphAnalyzer()
        self.pos2n = {None: 0}
        self.n2pos = [None, ]
        self.train_size = train_size
        self.seed = seed
        self.model = CatBoostClassifier(loss_function="Logloss",
                                   eval_metric='Accuracy',
                                   use_best_model=True, random_seed=self.seed)
        super().__init__(seed)

    def get_placeholder(self, token):
        if len(token) < 5 and token[0] == '(' and token[-1] == ')':
            return token[1:-1]
        return ''

    def save(self, path="data/models/solver17.pkl"):
        joblib.dump(self.model, path)

    def load(self, path="data/models/solver17.pkl"):
        self.model = joblib.load(path)

    def get_target(self, task):
        if 'solution' not in task:
            return []
        y_true = task['solution']['correct_variants'] if 'correct_variants' in task['solution'] \
            else [task['solution']['correct']]
        return list(y_true[0])

    def clear_token(self, token):
        for char in '?.!/;:':
            token = token.replace(char, '')
        return token

    def get_feat(self, token):
        if self.get_placeholder(token):
            return 'PHDR'
        else:
            p = self.morph.parse(self.clear_token(token))[0]
            return str(p.tag.POS)

    def encode_feats(self, feats):
        res = []
        for feat in feats:
            if feat not in self.pos2n and self.is_train_task:
                self.pos2n[feat] = len(self.pos2n)
                self.n2pos.append(feat)
            elif feat not in self.pos2n:
                feat = None
            res.append(self.pos2n[feat])
        return res

    def correct_spaces(self, text):
        text = re.sub(r'(\(\d\))', r' \1 ', text)
        text = re.sub(r'  *', r' ', text)
        return text

    def parse_task(self, task):
        feat_ids = [-3, -2, -1, 1, 2, 3]
        tokens = self.correct_spaces(task['text']).split()
        targets = self.get_target(task)
        X, y = [], []
        for i, token in enumerate(tokens):
            placeholder = self.get_placeholder(token)
            if not placeholder:
                continue
            if placeholder in targets:
                y.append(1)
            else:
                y.append(0)
            feats = []
            for feat_idx in feat_ids:
                if i + feat_idx < 0 or i + feat_idx >= len(tokens):
                    feats.append('PAD')
                else:
                    feats.append(self.get_feat(tokens[i + feat_idx]))
            X.append(self.encode_feats(feats))
        return X, y

    def fit(self, tasks):
        self.is_train_task = True
        X, y = [], []
        for task in tasks:
            task_x, task_y = self.parse_task(task)
            X += task_x
            y += task_y
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, y, shuffle=True,
                                                          train_size=self.train_size, random_state=self.seed)
        cat_features = [0, 1, 2, 3, 4, 5]
        self.model = self.model.fit(X_train, Y_train, cat_features, eval_set=(X_dev, Y_dev))
        self.has_model = True

    def predict_from_model(self, task):
        self.is_train_task = False
        X, y = self.parse_task(task)
        pred = self.model.predict(X)
        pred = [str(i+1) for i, p in enumerate(pred) if p >= 0.5]
        return pred if pred else ["1"]
