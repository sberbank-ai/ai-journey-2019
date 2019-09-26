import operator
import random
import re
from sklearn.metrics.pairwise import cosine_similarity
from solvers.utils import BertEmbedder


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.has_model = True

    def init_seed(self):
        random.seed(self.seed)

    def compare_text_with_variants(self, text, variants, label, num=2):
        text_choices = [re.sub("[0-9]\\)\s?", "", choice["text"]) for choice in variants ]
        # CHECK MAX OF TEXT TO EMBEDDER
        # token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
        # ending = " ".join(token_list[:512][-5:]).replace(" ##", "").replace('##', "")
        # text = "".join(text.split(ending)[0])

        text_vector = self.sentence_embedding([text[:1100]])
        variant_vectors = self.sentence_embedding(text_choices)
        
        predictions = {}
        i = 0
        for j in variant_vectors:
            sim = cosine_similarity(text_vector[0].reshape(1, -1), j.reshape(1, -1)).flatten()[0]
            predictions[i] = sim
            i += 1

        if label == 'pro':
            indexes = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)[:num]
        else:
            indexes = sorted(predictions.items(), key=operator.itemgetter(1), reverse=True)[-num:]

        return sorted([str(i[0] + 1) for i in indexes])

    def fit(self, tasks):
        pass
    
    def load(self, path='data/models/solver22.pkl'):
        pass

    def save(self, path='data/models/solver22.pkl'):
        pass


    def parse_task(self, task):
        assert task["question"]["type"] == "multiple_choice"

        choices = task["question"]["choices"]
        description = task["text"]

        label = "pro" # against
        if "противоречат" in description:
            label = "against"
        elif "не соответствуют" in description:
            label = "against"

        bad_strings = ["Какие из высказываний соответствуют содержанию текста?",
                       "Какие из высказываний не соответствуют содержанию текста?",
                       "Укажите номера ответов.", "Какие из высказываний противоречат содержанию текста?"
                       "Источник текста не определён."]
        if "(1)" in description:
            text = "(1)" + " ".join(description.split("(1)")[1:])
        else:
            text = "(1)" + " ".join(description.split("1)")[1:])
        for bad in bad_strings:
            text = re.sub(bad, "", text)

        return text, choices, label

    def predict(self, task):
        if not self.has_model:
            return self.predict_random(task)
        else:
            return self.predict_from_model(task)

    def predict_from_model(self, task):
        text, choices, label = self.parse_task(task)
        result = self.compare_text_with_variants(text, choices, label)
        return result


    def predict_random(self, task):
        choices = task["question"]["choices"]
        pred = []
        for _ in range(random.choice([2, 3])):
            choice = random.choice(choices)
            pred.append(choice["id"])
            choices.remove(choice)
        return pred

