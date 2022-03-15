import collections
import os
import pickle
import random

import numpy as np


def tokenize_string(sample):
    return tuple(sample.lower().split(' '))


class Dataset:
    def __init__(self, train_path, test_path, max_length, max_train_size=0, max_test_size=0, skip_unk=False):
        self.train_path = train_path
        self.test_path = test_path
        self.max_length = max_length
        self.max_train_size = max_train_size
        self.max_test_size = max_test_size
        self.skip_unk = skip_unk

        self.train_passwords = list()
        self.test_passwords = set()
        self.charmap = {}
        self.inv_charmap = []

        if not self.load():
            self.load_dataset(is_train=True)
            self.load_dataset(is_train=False)

            train_set = set(self.train_passwords)
            intersection = train_set & self.test_passwords
            print('Train/test intersection {}'.format(len(intersection)))

            self.test_passwords = self.test_passwords - intersection
            print('Clean test set size: {}'.format(len(self.test_passwords)))

        self.charmap_size = len(self.charmap)
        self.train_passwords_size = len(self.train_passwords)
        self.test_passwords_size = len(self.test_passwords)

        print(f'train {self.train_passwords_size} test {self.test_passwords_size}')
        self.save()

    def get_train_size(self):
        return self.train_passwords_size

    def get_test_size(self):
        return self.test_passwords_size

    def get_charmap_size(self):
        return self.charmap_size

    def get_dataset_filename(self):
        return 'dataset_{}_{:.2e}_{:.2e}.pk'.format(self.max_length, self.max_train_size, self.max_test_size)

    def load(self):
        full_path = os.path.join(os.path.split(self.train_path)[0], self.get_dataset_filename())
        if os.path.exists(full_path):
            print(f'Loading dataset {full_path} from pickle.')
            with open(full_path, 'rb') as fin:
                loaded = pickle.load(fin)
            if loaded['max_length'] == self.max_length and loaded['max_train_size'] == self.max_train_size \
                    and loaded['max_test_size'] == self.max_test_size:
                self.__dict__.update(loaded)
                print(f'Loaded dataset {full_path} from pickle.')
                return True

            print(f'{full_path} saved on disk has different parameters from current run. Rebuilding')
        return False

    def save(self):
        full_path = os.path.join(os.path.split(self.train_path)[0], self.get_dataset_filename())
        with open(full_path, 'wb') as fout:
            pickle.dump(self.__dict__, fout)
            print('Pickled dataset saved')

    def load_dataset(self, tokenize=False, max_vocab_size=2048, is_train=True):
        lines = []

        with open(self.train_path if is_train else self.test_path, 'r') as f:
            for line in f:
                line = line[:-1]
                if tokenize:
                    line = tokenize_string(line)
                else:
                    line = tuple(line)

                if len(line) > self.max_length:
                    continue  # don't include this sample, its too long

                # right pad with ` character
                lines.append(self.pad_password(line))

        if self.max_train_size != 0 and is_train:
            # keep seed fixed through different runs
            prng = random.Random(42)
            lines = prng.sample(lines, self.max_train_size)

        np.random.shuffle(lines)

        counts = collections.Counter(char for line in lines for char in line)

        if is_train:
            self.charmap = {}
            self.inv_charmap = []
            if not self.skip_unk:
                self.charmap['unk'] = 0
                self.inv_charmap.append('unk')

            for char, count in counts.most_common(max_vocab_size - 1):
                if char not in self.charmap:
                    self.charmap[char] = len(self.inv_charmap)
                    self.inv_charmap.append(char)

        passwords = []
        for line in lines:
            filtered_line = []
            for char in line:
                if char in self.charmap:
                    filtered_line.append(char)
                else:  # this condition should never be triggered
                    if self.skip_unk:
                        filtered_line = None
                        break
                    filtered_line.append('unk')
            if filtered_line is not None:
                passwords.append(tuple(filtered_line))
        if is_train:
            # We need duplicates during training
            self.train_passwords = [tuple(self.encode_password(pwd)) for pwd in passwords]
        else:
            # Here we use set to remove duplicates
            self.test_passwords = set([tuple(self.encode_password(pwd)) for pwd in passwords])
            if self.max_test_size and self.max_test_size < len(self.test_passwords):
                self.test_passwords = set(random.sample(set([tuple(self.encode_password(pwd)) for pwd in passwords]),
                                                        self.max_test_size))
                passwords = self.test_passwords

        print('{} set: loaded {} out of {} lines in dataset. {} filtered'
              .format('Training' if is_train else 'Test', len(passwords), len(lines), len(lines) - len(passwords)))

    def pad_password(self, password):
        return password + (("`",) * (self.max_length - len(password)))

    def encode_password(self, padded_password):
        return [self.charmap[c] for c in padded_password]

    def decode_password(self, encoded_password):
        return tuple([self.inv_charmap[c] if 0 <= c < self.charmap_size else c for c in encoded_password])

    def get_batches(self, batch_size=128, is_train=True):
        data = self.train_passwords if is_train else self.test_passwords

        np.random.shuffle(data)

        for i in range(0, len(data) - batch_size + 1, batch_size):
            yield np.array([np.array(pwd) for pwd in data[i:i + batch_size]], dtype='float32')
