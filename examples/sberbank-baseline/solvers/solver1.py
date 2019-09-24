import re
import random
import operator
import pymorphy2
from nltk.tokenize import ToktokTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from solvers.utils import BertEmbedder


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.is_train_task = False
        self.morph = pymorphy2.MorphAnalyzer()
        self.toktok = ToktokTokenizer()
        self.seed = seed
        self.init_seed()

    def init_seed(self):
        random.seed(self.seed)

    def predict(self, task):
        return self.predict_from_model(task)

    def get_num(self, text):
        lemmas = [self.morph.parse(word)[0].normal_form for word in self.toktok.tokenize(text)]
        if 'указывать' in lemmas and 'предложение' in lemmas:
            w = lemmas[lemmas.index('указывать') + 1]  # first
            d = {'один': 1,
                 'два': 2,
                 'три': 3,
                 'четыре': 4,
                 'предложение': 1}
            if w in d:
                return d[w]
        elif 'указывать' in lemmas and 'вариант' in lemmas:
            return 'unknown'
        return 1

    def compare_text_with_variants(self, text, variants, num=1):
        text_vector = self.sentence_embedding([text])
        variant_vectors = self.sentence_embedding(variants)
        i, predictions = 0, {}
        for j in variant_vectors:
            sim = cosine_similarity(text_vector[0].reshape(1, -1), j.reshape(1, -1)).flatten()[0]
            predictions[i] = sim
            i += 1
        indexes = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)[:num]
        return sorted([str(i[0] + 1) for i in indexes])

    def sent_split(self, text):
        reg = r'\(*\d+\)'
        return re.split(reg, text)

    def process_task(self, task):
        first_phrase, task_text = re.split(r'\(*1\)', task['text'])[:2]
        variants = [t['text'] for t in task['question']['choices']]
        text, task = "", ""
        if 'Укажите' in task_text:
            text, task = re.split('Укажите ', task_text)
            task = 'Укажите ' + task
        elif 'Укажите' in first_phrase:
            text, task = task_text, first_phrase
        return text, task, variants

    def fit(self, tasks):
        pass

    def load(self, path=""):
        pass
    
    def save(self, path=''):
        pass

    def predict_from_model(self, task, num=2):
        text, task, variants = self.process_task(task)
        result = self.compare_text_with_variants(text, variants, num=num)
        return result
