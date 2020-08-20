import numpy as np
from sklearn.svm import LinearSVC, SVC 

texts = dict()
labels = dict()
for tag in ['train', 'valid', 'test']:
    with open(f'{tag}.npy', 'rb') as f:
        texts[tag] = np.load(f)
    with open(f'{tag}_labels.npy', 'rb') as f:
        labels[tag] = np.load(f)

for C in 10. ** np.arange(3, 5):
    print(f'# C={C}:', end=' ', flush=True)
    svm = SVC(C=C)
    svm.fit(texts['train'], labels['train'])
    print(f"{svm.score(texts['valid'], labels['valid'])}")
