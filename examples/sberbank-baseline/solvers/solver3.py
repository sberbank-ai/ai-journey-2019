import re
import operator
import random
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

    def clean_text(self, text):
        newtext, logic = [], ["PREP", "CONJ", "Apro", "PRCL", "INFN", "VERB", "ADVB"]
        for token in self.toktok.tokenize(text):
            if any(tag in self.morph.parse(token)[0].tag for tag in logic):
                newtext.append(self.morph.parse(token)[0].normal_form)
        return ' '.join(newtext)

    def get_pos(self, text):
        pos, lemmas = 'word', [self.morph.parse(word)[0].normal_form for word in
                  self.toktok.tokenize(text)]
        if 'сочинительный' in lemmas:
            pos = "CCONJ"
        elif 'подчинительный' in lemmas:
            pos = "SCONJ"
        elif 'наречие' in lemmas:
            pos = "ADV"
        elif 'союзный' in lemmas:
            pos = "ADVPRO"
        elif 'местоимение' in lemmas:
            pos = "PRO"
        elif 'частица' in lemmas:
            pos = "PART"
        return pos

    def get_num(self, text):
        lemmas = [self.morph.parse(word)[0].normal_form for word in
                  self.toktok.tokenize(text)]
        if 'слово' in lemmas and 'предложение' in lemmas:
            d = {'один': 1,
                 'два': 2,
                 'три': 3,
                 'четыре': 4,
                 'первый': 1,
                 'второй': 2,
                 'третий': 3,
                 'четвертый': 4,
                 }
            for i in lemmas:
                if i in d:
                    return d[i]
        return 1

    def sent_split(self, text):
        reg = r'\(\n*\d+\n*\)'
        return re.split(reg, text)

    def compare_text_with_variants(self, word, text, variants):
        sents = self.sent_split(text)
        for sent in sents:
            lemmas = [self.morph.parse(word)[0].normal_form for word in
                  self.toktok.tokenize(text)]
            if word.lower() in lemmas:
                text = sent
        text_vector = self.sentence_embedding([text])
        variant_vectors = self.sentence_embedding(variants)
        i, predictions = 0, {}
        for j in variant_vectors:
            sim = cosine_similarity(text_vector[0].reshape(1, -1), j.reshape(1, -1)).flatten()[0]
            predictions[i] = sim
            i += 1
        indexes = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)[:1]
        return sorted([str(i[0] + 1) for i in indexes])

    def process_task(self, task):
        try:
            first_phrase, task_text = re.split(r'\(\n*1\n*\)', task['text'])
        except ValueError:
            first_phrase, task_text = ' '.join(re.split(r'\(\n*1\n*\)', task['text'])[:-1]), \
                                    re.split(r'\(\n*1\n*\)', task['text'])[-1]
        variants = [t['text'] for t in task['question']['choices']]
        text, task, word = "", "", ""
        if 'Определите' in task_text:
            text, task = re.split('Определите', task_text)
            task = 'Определите ' + task
            word = re.split('\.', re.split('значения слова ', text)[1])[0]
        elif 'Определите' in first_phrase:
            text, task = task_text, first_phrase
            word = re.split('\.', re.split('значения слова ', task)[1])[0]
        return text, task, variants, word

    def fit(self, tasks):
        pass

    def load(self, path="data/models/solver3.pkl"):
        pass
    
    def save(self, path='data/models/solver3.pkl'):
        pass
    
    def predict_from_model(self, task):
        text, task, variants, word = self.process_task(task)
        result = self.compare_text_with_variants(word, text, variants)
        return result
