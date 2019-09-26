from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
from itertools import combinations
from keras.preprocessing.sequence import pad_sequences
from solvers.utils import BertEmbedder, singleton
import re
import numpy as np
import random
import gensim
import time
import os


class SiameseBiLSTM(BertEmbedder):

    def __init__(self, model=None, embedding_dim=768, max_sequence_length=40, number_lstm=50, number_dense=50, rate_drop_lstm=0.17,
                 rate_drop_dense=0.25, hidden_activation='relu', validation_split_ratio=0.2):
        super(SiameseBiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.number_lstm_units = number_lstm
        self.rate_drop_lstm = rate_drop_lstm
        self.number_dense_units = number_dense
        self.activation_function = hidden_activation
        self.rate_drop_dense = rate_drop_dense
        self.validation_split_ratio = validation_split_ratio

    def train_model(self, sentences_pairs, is_similar, model_save_directory='./'):
        """
        Train Siamese network to find similarity between sentences in `sentences_pair`
            Steps Involved:
                1. Pass the each from sentences_pairs  to bidirectional LSTM encoder.
                2. Merge the vectors from LSTM encodes and passed to dense layer.
                3. Pass the  dense layer vectors to sigmoid output layer.
                4. Use cross entropy loss to train weights
        Args:
            sentences_pair (list): list of tuple of sentence pairs
            is_similar (list): target value 1 if same sentences pair are similar otherwise 0
        Returns:
            return (best_model_path):  path of best model
        """
        train_data_x1, train_data_x2, train_labels, leaks_train, \
        val_data_x1, val_data_x2, val_labels, leaks_val = self.create_train_dev_set(
            sentences_pairs, is_similar, self.max_sequence_length,
            self.validation_split_ratio
        )

        embedding_layer = Embedding(119547, 768, weights=[self.embedding_matrix], trainable=False)

        lstm_layer = Bidirectional(
            LSTM(self.number_lstm_units, dropout=self.rate_drop_lstm, recurrent_dropout=self.rate_drop_lstm)
        )

        sequence_1_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_1 = embedding_layer(sequence_1_input)
        x1 = lstm_layer(embedded_sequences_1)

        sequence_2_input = Input(shape=(self.max_sequence_length,), dtype='int32')
        embedded_sequences_2 = embedding_layer(sequence_2_input)
        x2 = lstm_layer(embedded_sequences_2)

        leaks_input = Input(shape=(leaks_train.shape[1],))
        leaks_dense = Dense(int(self.number_dense_units/2), activation=self.activation_function)(leaks_input)

        merged = concatenate([x1, x2, leaks_dense])
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        merged = Dense(self.number_dense_units, activation=self.activation_function)(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(self.rate_drop_dense)(merged)
        preds = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[sequence_1_input, sequence_2_input, leaks_input], outputs=preds)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=3)

        STAMP = 'lstm_%d_%d_%.2f_%.2f' % (self.number_lstm_units, self.number_dense_units, self.rate_drop_lstm, self.rate_drop_dense)

        checkpoint_dir = model_save_directory + 'checkpoints/' + str(int(time.time())) + '/'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        bst_model_path = checkpoint_dir + STAMP + '.h5'

        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)

        tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

        model.fit([train_data_x1, train_data_x2, leaks_train], train_labels,
                  validation_data=([val_data_x1, val_data_x2, leaks_val], val_labels),
                  epochs=50, batch_size=64, shuffle=True,
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        return bst_model_path


    def preprocess_strings(self, some_queries):

        id_text_tr = []
        for text in some_queries:
            try:
                token_list = self.tokenizer.tokenize("[CLS] " + text + " [SEP]")
                segments_ids, indexed_tokens = [1] * len(token_list), self.tokenizer.convert_tokens_to_ids(token_list)
                id_text_tr.append(indexed_tokens)
            except KeyError:
                continue
        return id_text_tr


    def create_train_dev_set(self, sentences_pairs, is_similar, max_sequence_length, validation_split_ratio):

        sentences1 = [x[0].lower() for x in sentences_pairs]
        sentences2 = [x[1].lower() for x in sentences_pairs]

        train_sequences_1 = self.preprocess_strings(sentences1)
        train_sequences_2 = self.preprocess_strings(sentences2)

        leaks = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                 for x1, x2 in zip(train_sequences_1, train_sequences_2)]

        train_padded_data_1 = pad_sequences(train_sequences_1, maxlen=max_sequence_length)
        train_padded_data_2 = pad_sequences(train_sequences_2, maxlen=max_sequence_length)

        train_labels = np.array(is_similar)
        leaks = np.array(leaks)

        shuffle_indices = np.random.permutation(np.arange(len(train_labels)))
        train_data_1_shuffled = train_padded_data_1[shuffle_indices]
        train_data_2_shuffled = train_padded_data_2[shuffle_indices]
        train_labels_shuffled = train_labels[shuffle_indices]
        leaks_shuffled = leaks[shuffle_indices]

        dev_idx = max(1, int(len(train_labels_shuffled) * validation_split_ratio))

        del train_padded_data_1
        del train_padded_data_2

        train_data_1, val_data_1 = train_data_1_shuffled[:-dev_idx], train_data_1_shuffled[-dev_idx:]
        train_data_2, val_data_2 = train_data_2_shuffled[:-dev_idx], train_data_2_shuffled[-dev_idx:]
        labels_train, labels_val = train_labels_shuffled[:-dev_idx], train_labels_shuffled[-dev_idx:]
        leaks_train, leaks_val = leaks_shuffled[:-dev_idx], leaks_shuffled[-dev_idx:]

        return train_data_1, train_data_2, labels_train, leaks_train, \
               val_data_1, val_data_2, labels_val, leaks_val


    def create_test_data(self, test_sentence_pairs, max_len):
        test_sentences1 = [x[0].lower() for x in test_sentence_pairs]
        test_sentences2 = [x[1].lower() for x in test_sentence_pairs]

        test_sequences_1 = self.preprocess_strings(test_sentences1)
        test_sequences_2 = self.preprocess_strings(test_sentences2)
        leaks_test = [[len(set(x1)), len(set(x2)), len(set(x1).intersection(x2))]
                      for x1, x2 in zip(test_sequences_1, test_sequences_2)]

        leaks_test = np.array(leaks_test)
        test_data_1 = pad_sequences(test_sequences_1, maxlen=max_len)
        test_data_2 = pad_sequences(test_sequences_2, maxlen=max_len)

        return test_data_1, test_data_2, leaks_test


class Solver():

    def __init__(self, seed=42, path_to_model="data/models/siameise_model.h5"):
        self.has_model = True
        self.siamese = SiameseBiLSTM()
        self.best_model_path = path_to_model
        self.seed = seed
        self.init_seed()

    def init_seed(self):
        return random.seed(self.seed)

    def parse_task(self, task):
        """ link multiple_choice """
        assert task["question"]["type"] == "multiple_choice"

        choices = task["question"]["choices"]

        links, label = [], ""
        description = task["text"]

        if "двоеточие" in description:
            label = "двоеточие"
        if "тире" in description:
            label = "тире"
        if "запят" in description:
            label = "запятая"

        m = re.findall("[0-9]\\)", description)
        for n, match in enumerate(m, 1):
            first, description = description.split(match)
            if len(first) > 1 and "Найдите" not in first:
                links.append(first)
                if n == len(m):
                    description = description.split('\n')[0]
                    links.append(description.replace(' (', ''))

        assert len(links) == len(choices)

        return links, label

    def dash_task(self, choices):
        hypothesys = []

        for choice in choices:
            if ' –' in choice:
                hypothesys.append(choice)
            if " —" in choice:
                hypothesys.append(choice)
        return hypothesys

    def semicolon_task(self, choices):
        hypothesys = []
        for choice in choices:
            if ':' in choice:
                hypothesys.append(choice)
        return hypothesys

    def comma_task(self, choices):
        hypothesys = []
        for choice in choices:
            if ', ' in choice:
                hypothesys.append(choice)
        return hypothesys

    def fit(self, tasks):
        self.load(path="data/models/siameise_model.h5")
        return

        sentences_pairs, is_similar_target = [], []

        for task in tasks:
            choices, label = self.parse_task(task)
            y_true = task['solution']['correct_variants'] if 'correct_variants' in task['solution'] else [
                task['solution']['correct']]

            pairs, indexes = [], []
            for y in y_true[0]:
                for n, choice in enumerate(choices, 1):
                    if int(y) == n:
                        pairs.append(choice)
                    else:
                        indexes.append(choice)

            good_pairs = list(combinations(pairs, 2))

            for pair in good_pairs:
                sentences_pairs.append(pair)
                is_similar_target.append(1)

            bad_pair = indexes[:2]
            sentences_pairs.append(bad_pair)
            is_similar_target.append(0)

        self.best_model_path = self.siamese.train_model(
            sentences_pairs, is_similar_target
        )
        return self.best_model_path

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

    def load(self, path="data/models/siameise_model.h5"):
        print("Hi!, It's load")
        self.best_model_path = "data/models/siameise_model.h5"
        self.siamese_model_loaded = self.get_model()
        print("Siamese model is loaded")
        print(self.siamese.embedding_matrix)
        return self.siamese_model_loaded

    def save(self, path='data/models/siameise_model.h5'):
        pass

    @singleton
    def get_model(self):
        model = load_model(self.best_model_path)
        return model

    def predict_from_model(self, task):
        test_sentence_pairs = []
        choices, label = self.parse_task(task)
        choices_dict = {}
        for n, choice in enumerate(choices, 1):
            choices_dict[choice] = n

        if label == 'тире':
            hypothesys = self.dash_task(choices)
        elif label == 'запятая':
            hypothesys = self.comma_task(choices)
        else:
            hypothesys = self.semicolon_task(choices)

        for pair in list(combinations(hypothesys, 2)):
            test_sentence_pairs.append(pair)

        test_data_x1, test_data_x2, leaks_test = self.siamese.create_test_data(test_sentence_pairs, 40)

        try:
            preds = list(self.siamese_model_loaded.predict([test_data_x1, test_data_x2, leaks_test], verbose=1).ravel())
        except ValueError:
            preds = []

        preds = [pr for pr in preds]

        final_answer = []
        for pair, pred in zip(test_sentence_pairs, preds):

            max_pred = max(preds)
            if pred == max_pred:
                for n, choice in enumerate(choices, 1):
                    if choice == pair[0]:
                        final_answer.append(str(n))
                    if choice == pair[1]:
                        final_answer.append(str(n))
        if final_answer:
            return final_answer
        else:
            ch = [str(n) for n, choice in enumerate(choices, 1)]
            random.shuffle(ch)
            return ch[:2]
