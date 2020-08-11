import random
import numpy as np
import thinc.extra.datasets
from fasttext_utils import *

train_and_valid_data, test_data = thinc.extra.datasets.imdb()

random.seed(123)
random.shuffle(train_and_valid_data)

train_and_valid_texts, train_and_valid_labels = zip(*train_and_valid_data)
test_texts, test_labels = zip(*test_data)

split = int(len(train_and_valid_data) * 0.8)
train_texts, train_labels = train_and_valid_texts[:split], train_and_valid_labels[:split]
valid_texts, valid_labels = train_and_valid_texts[split:], train_and_valid_labels[split:]

print(f'# {len(train_texts)} train, {len(valid_texts)} validation, {len(test_texts)} test docs')

texts = { 'train': train_texts, 'valid': valid_texts, 'test': test_texts }
labels = { 'train': train_labels, 'valid': valid_labels, 'test': test_labels }

for k in texts:
    X = list()
    for text in texts[k]:
        vec = get_sentence_vector(text)
        X.append(vec)
    X = np.array(X)
    with open(f'{k}.npy', 'wb') as f:
        np.save(f, X)
