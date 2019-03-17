import re
import string
import unidecode
import numpy as np
import scipy.sparse as sp
import emoji
from sklearn import utils as skutils
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion


class RemoveConsecutiveSpaces(BaseEstimator, TransformerMixin):
    def remove_consecutive_spaces(self, s):
        return ' '.join(s.split())

    def transform(self, x):
        return [self.remove_consecutive_spaces(s) for s in x]

    def fit(self, x, y=None):
        return self


class RemovePunct(BaseEstimator, TransformerMixin):
    non_special_chars = re.compile('[^A-Za-z0-9 ]+')

    def remove_punct(self, s):
        return re.sub(self.non_special_chars, '', s)

    def transform(self, x):
        return [self.remove_punct(s) for s in x]

    def fit(self, x, y=None):
        return self


class Lowercase(BaseEstimator, TransformerMixin):
    def transform(self, x):
        return [s.lower() for s in x]

    def fit(self, x, y=None):
        return self


class RemoveTone(BaseEstimator, TransformerMixin):
    def remove_tone(self, s):
        return unidecode.unidecode(s)

    def transform(self, x):
        return [self.remove_tone(s) for s in x]

    def fit(self, x, y=None):
        return self


class NumWordsCharsFeature(BaseEstimator, TransformerMixin):
    def count_char(self, s):
        return len(s)

    def count_word(self, s):
        return len(s.split())

    def transform(self, x):
        count_chars = sp.csr_matrix([self.count_char(s) for s in x], dtype=np.float64).transpose()
        count_words = sp.csr_matrix([self.count_word(s) for s in x], dtype=np.float64).transpose()

        return sp.hstack([count_chars, count_words])

    def fit(self, x, y=None):
        return self


class ExclamationMarkFeature(BaseEstimator, TransformerMixin):
    def count_exclamation(self, s):
        count = s.count('!') + s.count('?')
        return count / (1 + len(s.split()))

    def transform(self, x):
        counts = [self.count_exclamation(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumCapitalLettersFeature(BaseEstimator, TransformerMixin):
    def count_upper(self, s):
        n_uppers = sum(1 for c in s if c.isupper())
        return n_uppers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_upper(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumLowercaseLettersFeature(BaseEstimator, TransformerMixin):
    def count_lower(self, s):
        n_lowers = sum(1 for c in s if c.islower())
        return n_lowers / (1 + len(s))

    def transform(self, x):
        counts = [self.count_lower(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumPunctsFeature(BaseEstimator, TransformerMixin):
    def count(self, l1, l2):
        return sum([1 for x in l1 if x in l2])

    def count_punct(self, s):
        n_puncts = self.count(s, set(string.punctuation))
        return n_puncts / (1 + len(s))

    def transform(self, x):
        counts = [self.count_punct(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


class NumEmojiFeature(BaseEstimator, TransformerMixin):
    def count_emoji(self, s):
        emoji_list = []
        for c in s:
            if c in emoji.UNICODE_EMOJI:
                emoji_list.append(c)
        return len(emoji_list) / (1 + len(s.split()))

    def transform(self, x):
        counts = [self.count_emoji(s) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).transpose()

    def fit(self, x, y=None):
        return self


def load_train_data(filename):
    data = []
    target = []
    target_names = ['pos', 'neg']

    with open(filename) as file:
        lines = file.read()

    samples = lines.strip().split("\n\ntrain_")

    for sample in samples:
        sample = sample.strip()
        d = sample[8:-3].strip()
        t = sample[-1:]
        data.append(d)
        target.append(t)

    return skutils.Bunch(data=data, target=target, target_names=target_names)


def load_test_data(filename):
    data = []

    with open(filename) as file:
        lines = file.read()

    samples = lines.strip().split("\n\ntest_")

    for sample in samples:
        sample = sample.strip()
        d = sample[8:-1].strip()
        data.append(d)

    return skutils.Bunch(data=data)


if __name__ == '__main__':
    train_data = load_train_data('train.crash')

    clf = Pipeline([
        ('remove_spaces', RemoveConsecutiveSpaces()),
        ('features', FeatureUnion([
            ('custom_features_pipeline', Pipeline([
                ('custom_features', FeatureUnion([
                    ('f01', NumWordsCharsFeature()),
                    ('f02', NumCapitalLettersFeature()),
                    ('f03', ExclamationMarkFeature()),
                    ('f04', NumPunctsFeature()),
                    ('f05', NumLowercaseLettersFeature()),
                    ('f06', NumEmojiFeature())
                ], n_jobs=-1)),
                ('scaler', StandardScaler(with_mean=False))
            ])),
            ('word_char_features_pipeline', Pipeline([
                ('lowercase', Lowercase()),
                ('word_char_features', FeatureUnion([
                    ('with_tone', Pipeline([
                        ('remove_punct', RemovePunct()),
                        ('tf_idf_word', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))
                    ])),
                    ('with_tone_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
                    ('with_tone_char_wb', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char_wb')),
                    ('without_tone', Pipeline([
                        ('remove_tone', RemoveTone()),
                        ('without_tone_features', FeatureUnion([
                            ('tf_idf', Pipeline([
                                ('remove_punct', RemovePunct()),
                                ('word', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))
                            ])),
                            ('tf_idf_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
                            ('tf_idf_char_wb',
                             TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char_wb'))
                        ], n_jobs=-1))
                    ]))
                ], n_jobs=-1))
            ]))
        ], n_jobs=-1)),
        ('alg', SVC(kernel='linear', C=0.2175, class_weight=None, verbose=True))
    ])

    clf.fit(train_data.data, train_data.target)

    test_data = load_test_data('test.crash')

    predicted = clf.predict(test_data.data)

    with open('submission.csv', 'w') as f:
        f.write('id,label\n')
        i = 0
        for p in predicted:
            f.write('test_{id},{p}\n'.format(id=str(i).zfill(6), p=p))
            i = i + 1
