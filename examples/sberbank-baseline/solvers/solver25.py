import re
import random
import pymorphy2
from nltk.tokenize import ToktokTokenizer


class Solver(object):

    def __init__(self, seed=42):
        self.is_train_task = False
        self.morph = pymorphy2.MorphAnalyzer()
        self.toktok = ToktokTokenizer()
        self.seed = seed
        self.init_seed()

    def init_seed(self):
        random.seed(self.seed)

    def lemmatize(self, text):
        return [self.morph.parse(word)[0].normal_form for word in
                self.toktok.tokenize(text.strip())]

    def predict(self, task):
        return self.predict_from_model(task)

    def get_word(self, text):
        try:
            return re.split('»', re.split('«', text)[1])[0]
        except:
            return ''

    def get_pos(self, text):
        pos = []
        lemmas = self.lemmatize(text)
        lemmas = [l for l in lemmas if l!=' ']
        if 'сочинительный' in lemmas:
            pos.append("CCONJ")
        if 'подчинительный' in lemmas:
            pos.append("SCONJ")
        if 'наречие' in lemmas:
            pos.append("ADV")
        if 'союзный' in lemmas:
            pos.append("ADVPRO")
        if 'частица' in lemmas:
            pos.append("PART")
        if 'определительный' in lemmas:
            pos.append("OPRO")
        if 'личный' in lemmas:
            pos.append("LPRO")
        if 'указательный' in lemmas:
            pos.append("UPRO")
        return pos
    
    def sent_split(self, text):
        reg = r'\(*\n*\d+\n*\)'
        return re.split(reg, text)

    def get_num(self, text):
        nums = 0
        res = re.search('\d+([–|-|—])*\d*', text)
        if res:
            res = res[0]
            if re.search(r'–|-|—', res):
                nums = re.split(r'–|-|—', res)
                nums = list(range(int(nums[0]), int(nums[1])+1))
            else:
                nums = [int(res)]
        return nums

    def compare_text_with_variants(self, pos, text, nums=[]):
        indexes = []
        sents = self.sent_split(text)
        dic = {"CCONJ":['но','а','и','да', 'тоже', 'также', 'зато', 'однако', 'же', 'или', 'либо'],
              "SCONJ":['если','хотя','однако','когда','что','потомучто'],
              "ADV":['сейчас','сегодня'],
              "ADVPRO":['который','которая'],
              "OPRO":['этот','это','эта','все', 'сам', 'самый', 'весь', 'всякий', 'каждый', 'любой', 'другой', 'иной', 'всяк', 'всяческий'],
               "LPRO":['я', 'ты', 'он', 'она', 'оно', 'мы', 'вы', 'они'],
               "UPRO":['этот','это','эта','все', 'тот', 'такой', 'таков', 'столько', 'сей', 'оный' ],
                "PART":['только','именно','не', 'ни', 'бы', 'лишь', 'пусть', 'дескать']
              }
        if not pos:
            return [str(random.choice(nums))]
        for s in nums:
            lemmas = self.lemmatize(sents[s-1])
            lemmas = [l for l in lemmas if l!=' ']
            conditions=0
            for p in pos:
                variants = dic[p]
                if sum([v in lemmas for v in variants]):
                    conditions+=1
            if conditions==len(pos):
                indexes.append(s)
        if not indexes:
            indexes = [random.choice(nums)]

        return [str(i) for i in sorted(indexes)]

    def eat_json(self, task):
        try:
            firstphrase, tasktext = re.split(r'\(\n*1\n*\)', task['text'])
        except ValueError:
            firstphrase, tasktext = ' '.join(re.split(r'\(\n*1\n*\)', task['text'])[:-1]),re.split(r'\(\n*1\n*\)', task['text'])[-1]
        if 'Среди предложений' in tasktext:
            text, task = re.split('Среди предложений', tasktext)
            task = 'Среди предложений '+task
            #word = re.split('\.', re.split('значения слова ', text)[1])[0]
        else:
            text, task = tasktext, firstphrase
            #word = re.split('\.', re.split('значения слова ', task)[1])[0]
        nums = self.get_num(task)
        pos = self.get_pos(task)
        return text, task, nums, pos

    def fit(self, tasks):
        pass

    def load(self, path='data/models/solver25.pkl'):
        pass

    def save(self, path='data/models/solver25.pkl'):
        pass

    def predict_from_model(self, task):
        text, task, nums, pos = self.eat_json(task)
        result = self.compare_text_with_variants(pos, text, nums=nums)
        return result
