import random
import numpy as np
import thinc.extra.datasets
import fasttext

model_path = 'cc.en.300.bin'
print(f'# loading {model_path} ...', flush=True) 
ft = fasttext.load_model(model_path)

train_and_valid_data, test_data = thinc.extra.datasets.imdb()

random.seed(123)
random.shuffle(train_and_valid_data)

train_and_valid_texts, train_and_valid_labels = zip(*train_and_valid_data)
test_texts, test_labels = zip(*test_data)

# manually split training and valid data sets
split = int(len(train_and_valid_data) * 0.8)
train_texts, train_labels = train_and_valid_texts[:split], train_and_valid_labels[:split]
valid_texts, valid_labels = train_and_valid_texts[split:], train_and_valid_labels[split:]

print(f'# {len(train_texts)} train, {len(valid_texts)} validation, {len(test_texts)} test docs')

splits = {
    'train': (train_texts, train_labels),
    'valid': (valid_texts, valid_labels),
    'test': (test_texts, test_labels)
}

for tag in splits:
    print(f'# {tag} set: ', end='', flush=True)
    cnt = 0
    X = list()
    for text in splits[tag][0]:
        vec = ft.get_sentence_vector(' '.join(text.split('\n')))
        X.append(vec)
        cnt += 1
        if cnt % 10000 == 0: print('*', end='', flush=True)
        elif cnt % 1000 == 0: print('-', end='', flush=True)
    X = np.array(X)
    with open(f'{tag}.npy', 'wb') as f:
        np.save(f, X, allow_pickle=False)
    with open(f'{tag}_labels.npy', 'wb') as f:
        np.save(f, np.array(splits[tag][1]), allow_pickle=False)
    print(flush=True)
