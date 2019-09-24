import os
import random
import re
import pymorphy2
from nltk.tokenize import ToktokTokenizer


class Solver(object):

    def __init__(self, seed=42, data_path = 'data/'):
        self.is_train_task = False
        self.morph = pymorphy2.MorphAnalyzer()
        self.toktok = ToktokTokenizer()
        self.seed = seed
        self.init_seed()
        self.synonyms = open(os.path.join(data_path, r'synonyms.txt'), 'r', encoding='utf8').readlines()
        self.synonyms = [re.sub('\.','', t.lower().strip('\n')).split(' ') for t in self.synonyms]
        self.synonyms = [[t for t in l if t]  for l in self.synonyms]
        self.antonyms = open(os.path.join(data_path, r'antonyms.txt'), 'r', encoding='utf8').readlines()
        self.antonyms = [t.strip(' \n').split(' - ') for t in self.antonyms]
        self.phraseology = open(os.path.join(data_path, r'phraseologs.txt'), 'r', encoding='utf8').readlines()
        self.phraseology = [[l for l in self.lemmatize(l) if l not in ['\n', ' ','...', '' ,',', '-', '.', '?',r' (', r'/']] for l in self.phraseology]

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
        if 'фразеологизм' in lemmas:
            pos = "PHR"
        elif 'синоним' in lemmas:
            pos = "SYN"
        elif 'антоним' in lemmas:
            pos = "ANT"
        elif 'антонимический' in lemmas:
            pos = "ANT"
        elif 'синонимический' in lemmas:
            pos = "SYN"
        else:
            pos = "DEF"
        return pos

    def full_intersection(self, small_lst, big_lst):
        if sum([value in big_lst for value in small_lst ] )==len(small_lst):
            return True
        return False

    def sent_split(self, text):
        reg = r'\(*\n*\d+\n*\)'
        return re.split(reg, text)

    def search(self, text_lemmas, lst):
        for l in lst:
            if self.full_intersection(l, text_lemmas):
                return ''.join(l)
        return ''

    def get_num(self, text):
        nums = 0
        res = re.search('\d+–*-*\d*', text)
        if res:
            res = res[0]
            if '–' in res:
                nums = res.split('–')
                nums = list(range(int(nums[0]), int(nums[1])+1))
            elif '-' in res:
                nums = res.split('-')
                nums = list(range(int(nums[0]), int(nums[1])+1))
            else:
                nums = [int(res)]
        return nums

    def compare_text_with_variants(self,pos, text, nums=[], word=''):
        indexes = []
        sents = self.sent_split(text)
        lemmas_all = []
        for s in nums:
            lemmas = self.lemmatize(sents[s-1])
            lemmas_all += [l for l in lemmas if l!=' ']
            conditions=0
        lemmas_all = [l for l in lemmas_all if re.match('\w+', l) and re.match('\w+', l)[0]==l]

        if pos=='SYN':
            variant = self.search(lemmas_all, self.synonyms)
        elif pos=='ANT':
            variant = self.search(lemmas_all, self.antonyms)
        else:
            variant = self.search(lemmas_all, self.phraseology)
        if variant:
            return variant
        else:
            return str(random.choice(lemmas_all))

    def eat_json(self, task):
        try:
            firstphrase, tasktext = re.split(r'\(\n*1\n*\)', task['text'])
        except ValueError:
            firstphrase, tasktext = ' '.join(re.split(r'\(\n*1\n*\)', task['text'])[:-1]),re.split(r'\(\n*1\n*\)', task['text'])[-1]
        if 'Из предложени' in tasktext:
            text, task = re.split('Из предложени', tasktext)
            task = 'Из предложени '+task
        else:
            text, task = tasktext, firstphrase
        nums = self.get_num(task)
        pos = self.get_pos(task)
        word = ''
        if pos=='DEF':
            word = self.get_word(task)
        return text, task, pos, nums, word


    def fit(self, tasks):
        pass

    def load(self, path='data/models/solver24.pkl'):
        pass

    def save(self, path='data/models/solver24.pkl'):
        pass

    def predict_from_model(self, task):
        text, task, pos, nums, word = self.eat_json(task)
        result = self.compare_text_with_variants(pos, text, nums=nums, word=word)
        return result
