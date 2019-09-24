import re
import pickle
import json
import joblib
import pymorphy2
import numpy as np
from summa import summarizer
from fastai.text import *
import pandas as pd
from fastai.callbacks import ReduceLROnPlateauCallback
import pandas as pd
import warnings
warnings.filterwarnings('ignore')



def clear(text):
    text = re.sub("[\t\r]+", "", text)
    text = re.sub("[ ]+[:]", ":", re.sub("[ ]+[.]", ".", re.sub("[«][ ]+", "«", re.sub("[ ]+[»]", "»", re.sub("[ ]+[,]", ",", re.sub("[ ]+", " ", text))))))
    text = re.sub("[ ]+[?]", "?", text)
    text = re.sub("[ ]+[!]", "!", text)
    text = re.sub("\n+", "\n", text)
    text = [line.strip() for line in text.split("\n")]
    # text = [line[1:] + line[1].upper() for line in text if len(line)]
    text = "\n".join(text)
    return text


def post_prc(learn, text, temperature, n_words=300, max_words=400):
    text = clear(text)
    while not text.endswith(tuple(".!?")) and n_words < max_words:
        n_words += 1
        text = clear(learn.predict(text, n_words=1, no_unk=True, temperature=temperature))

    return clear(text.replace("xxbos", " "))

def rus_tok(text, m = pymorphy2.MorphAnalyzer()):
        reg = '([0-9]|\W|[a-zA-Z])'
        toks = text.split()
        return [m.parse(t)[0].normal_form for t in toks if not re.match(reg, t)]

class Solver(object):
    """
    Note, fastai==1.0.52.

    Простой генератор на основе Ulmfit (https://github.com/mamamot/Russian-ULMFit) и тематического моделирования на текстах заданий.
    Дообучается на сочинениях (не учитывает ничего про условие задания).
    Генерация начинается с первой фразы, которая соответствует темам,
    которые были получены в ходе тематического моделирования текста задания.

    В код не включено обучение тематической модели. Интерпретация проведена в ручную.
    Первые фразы сочинений написаны вручную.
    
    Parameters
    ----------
    seed : path2config, str
        Path to config.
    model_name_to_save : str
        Model name for load pretrained ulmfit model and store this.
    dict_name_to_save : str
        Dict name for load pretrained ulmfit dict and store this.
    tf_vectorizer_path : str
        Path to vectorizer for topic modeling.
    lda_path : str
        Path to topic model.
    topics_path : str
        Path to topics with first phrases.
    is_load : bool, optional(default=True)
        Load or not pretrained models.

    Examples
    --------
    >>> # Basic usage
    >>> from .generator import Solver
    >>> g = Solver("path2config.json", "lm_5_ep_lr2-3_5_stlr", "itos",
    >>>     "tfvect.joblib", "lda.joblib", "topics.csv", is_load=False)
    >>> g = g.fit(df_path="10000.csv", num_epochs=5)
    >>> # Generate for task (you should pass task description)
    >>> text = gg.generate("Печенье и судьба")
    >>> # Save Generator
    >>> g.save()
    >>> # Load Generator
    >>> g = Solver("path2config.json")

    """
    def __init__(
        self, path2config, model_name_to_save=None, dict_name_to_save=None,
        tf_vectorizer_path=None, lda_path=None, topics_path=None, is_load=True, seed=42):
        self.path2config = path2config
        self.model_name_to_save = model_name_to_save
        self.dict_name_to_save = dict_name_to_save
        self.path2config = path2config
        self.data = None
        self.learn = None
        self.tf_vectorizer_path = tf_vectorizer_path
        self.lda_path = lda_path
        self.topics_path = topics_path
        self.tf_vectorizer = None
        self.lda = None
        self.topics = None
        self.topic_dic = None
        if is_load:
            self.load(path2config)
        self.ranker = self.TextRankSummarizer()
        self.seed = seed
        self.init_seed()

    def init_seed(self):
        random.seed(self.seed)
        
    def eat_json(self, task):
        if 'text' in task:
            text = re.split(r'\(*1\)', task['text'])[1]
            return re.sub('\(*\d+\)*', ' ', text)
        else:
            return ''
        
    def predict(self, task):
        task = self.eat_json(task)
        #print(task)
        return self.generate(input_task=task)

    class TextRankSummarizer(object):

        def __init__(self, language="russian", ratio=0.15, sentence_number=3):
            self.language = language
            self.ratio = ratio
            self.sentence_number = sentence_number

        def template_sentences(self, text_rank_summary):
            temp_sentences, sent_lengths = [], [len(sent.split()) for sent in text_rank_summary]
            for num in range(self.sentence_number):
                max_sent_length = int(np.argmax(sent_lengths))
                temp_sentences.append(text_rank_summary[max_sent_length])
                text_rank_summary.pop(max_sent_length)
                sent_lengths.pop(max_sent_length)
            return temp_sentences

        def second_paragraph(self, task_text):
            summary = summarizer.summarize(task_text, language=self.language, ratio=self.ratio, split=True)
            try:
                sentences = self.template_sentences(summary)
                first_sent = 'Автор иллюстрирует данную проблему на примере предложений "{}" и "{}".'.format(sentences[0], sentences[1])
                second_sent = 'На мой взгляд, читатель наблюдает авторскую позицию в предложении: "{}"'.format(sentences[2])
                paragraph = " ".join([first_sent, second_sent])
                return paragraph
            except:
                return ''


    def init_args(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def get_topic(self, documents):
        tf = self.tf_vectorizer.transform(documents)
        lda_doc_topic = self.lda.transform(tf)
        doc_topics = []
        for n in range(lda_doc_topic.shape[0]):
            topic_most_pr = lda_doc_topic[n].argmax()
            doc_topics.append(topic_most_pr)
        return [self.topic_dic[i] for i in doc_topics]

    def getinfo(self, topic):
        dic = {}
        for i in range(len(self.topics)):
            if self.topics.iloc[i]['Topic'] == topic:
                dic['Первая_фраза'] = self.topics.iloc[i]['First']
                dic['Произведения для аргументов'] = self.topics.iloc[i]['Books']
                dic['Тема'] = self.topics.iloc[i]['Theme']
                dic['Писатели'] = self.topics.iloc[i]['Authors']
        return dic

    def fit(self, df_path, model_name="lm_5_ep_lr2-3_5_stlr", dict_name="itos", num_epochs=5, is_fit_topics=False):
        df = pd.read_csv(df_path, sep="\t")
        bs = 16
        texts = pd.DataFrame(list(df.text))
        self.data = TextList.from_df(texts, 
                        processor=[TokenizeProcessor(tokenizer=Tokenizer(lang="xx")), 
                                                     NumericalizeProcessor(vocab=Vocab.load("models/{}.pkl".format(dict_name)))]).\
                random_split_by_pct(.1).\
                label_for_lm().\
                databunch(bs=bs)
        self.learn = language_model_learner(self.data, AWD_LSTM, pretrained=False, drop_mult=0.7, pretrained_fnames=[model_name, dict_name])
        self.learn.unfreeze()
        self.learn.lr_find(start_lr = slice(10e-7, 10e-5), end_lr=slice(0.4, 10))
        _ = self.learn.recorder.plot(skip_end=10, suggestion=True)
        best_lm_lr = self.learn.recorder.min_grad_lr
        #print(best_lm_lr)
        self.learn.fit_one_cycle(
            num_epochs, best_lm_lr, callbacks=[ReduceLROnPlateauCallback(self.learn, factor=0.8)])
        # TODO: fit lda
        if is_fit_topics:
            pass
        return self

    def save(obj):
        with open(obj.path2config, "w", encoding="utf-8") as file:
            json.dump({
                "model_name_to_save": obj.model_name_to_save,
                "dict_name_to_save": obj.dict_name_to_save,
                "tf_vectorizer_path": obj.tf_vectorizer_path,
                "lda_path": obj.lda_path,
                "topics_path": obj.topics_path
            }, file)
        obj.learn.save(obj.model_name_to_save)
        obj.learn.save_encoder(obj.model_name_to_save + "_enc")

    def load(self, path2config):
        with open(path2config, "r", encoding="utf-8") as file:
            config = json.load(file)
        bs = 16
        self.init_args(**config)
        # self = cls(path2config, is_load=False, **config)
        self.tf_vectorizer = joblib.load(self.tf_vectorizer_path)
        self.lda = joblib.load(self.lda_path)
        self.topics = pd.read_csv(self.topics_path, sep="\t")
        self.topic_dic = {int(i):self.topics.iloc[i]['Topic'] for i in range(len(self.topics))}

        self.data = TextList.from_df(pd.DataFrame(["tmp", "tmp"]), 
                        processor=[TokenizeProcessor(tokenizer=Tokenizer(lang="xx")), 
                                                     NumericalizeProcessor(
                                                         vocab=Vocab.load("data/generation/models/{}.pkl".format(self.dict_name_to_save)))]).\
                random_split_by_pct(.1).\
                label_for_lm().\
                databunch(bs=bs)

        self.dict_name_to_save = os.path.join('data/generation/models',self.dict_name_to_save )
        self.model_name_to_save = os.path.join('data/generation/models',self.model_name_to_save )
        self.learn = language_model_learner(self.data, AWD_LSTM, pretrained=False, drop_mult=0.7,
                                            pretrained_fnames=[self.model_name_to_save, self.dict_name_to_save])
        return self

    def generate(self, input_task="", seed="", n_words=300, no_unk=True, temperature=0.9, max_words=400):
        if input_task != "":
            seed = self.getinfo(self.get_topic([input_task])[0])['Первая_фраза']
        essay = post_prc(self.learn, self.learn.predict(seed, n_words=n_words, no_unk=no_unk, temperature=temperature), temperature, n_words, max_words)
        new_par = self.ranker.second_paragraph(input_task)
        essay = essay.split('\n')
        return '\n'.join([essay[0],new_par] +essay[1:])