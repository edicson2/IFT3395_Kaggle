# Latent Semantic Analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import matplotlib.pyplot as plt

vectorizer = TfidfVectorizer(stop_words='english', 
                             strip_accents='ascii', 
                             ngram_range=(1,2))
raw_texts = pd.read_csv('train.csv', index_col=0, header=0).Abstract
X_vect = vectorizer.fit_transform(raw_texts)
train = pd.read_csv('train.csv', index_col=0, header=0)
classes = list(set(train.Category))
y_train = pd.read_csv('train.csv', index_col=0, header=0).Category.apply(lambda text_label: classes.index(text_label))

# Dimensionality reduction
pca = TruncatedSVD(n_components=500)
x_pca = pca.fit_transform(X_vect)

# Data splitting
b = int(0.8 * x_pca.shape[0])
a = int(0.8 * b)
train_set = np.array(x_pca[:a])
valid_set = np.array(x_pca[a:b])
test_set = np.array(x_pca[b:])
train_label = np.array(y_train[:a])
valid_label = np.array(y_train[a:b])
test_label = np.array(y_train[b:])
tv_set = np.concatenate((train_set, valid_set), axis=0)
tv_label = np.concatenate((train_label, valid_label), axis=0)

# Model with default hyperparameters
classifier = SVC()
classifier.fit(tv_set, tv_label)

# evaluate model
print(accuracy_score(classifier.predict(test_set), test_label))

# output prediction
test_texts = pd.read_csv('test.csv', index_col=0, header=0).Abstract
X_test_vect = vectorizer.transform(test_texts)
X_test_pca = pca.transform(X_test_vect)
prediction = classifier.predict(X_test_pca)

# Exporting results
with open('submission.csv', 'w') as f:
    out = 'Id,Category'
    for i, result in enumerate(prediction):
        out += f"\n{str(i)},{classes[result]}"
    f.write(out)

# grid search
default = 1/(500*tv_set.var()) # 5.53
gamma_table = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, default, 10, 20]
C_table = [0.01, 0.1, 0.5, 1, 1.5, 2, 5, 10, 20]
accuracies = np.empty((len(gamma_table), len(C_table)))
for i, gamma in enumerate(gamma_table, position=0, leave=True):
    for j, C in enumerate(C_table):
        classifier = SVC(gamma=gamma, C=C)
        classifier.fit(tv_set, tv_label)
        accuracies[i][j] = accuracy_score(classifier.predict(test_set), test_label)

plt.imshow(accuracies, cmap='hot')
ax = plt.gca()
ax.set_xticks(np.arange(0, 9, 1));
ax.set_yticks(np.arange(0, 10, 1));
ax.set_xticklabels(C_table);
ax.set_yticklabels(gamma_table[:7] + ["default"] + gamma_table[8:]);
plt.ylabel("gamma")
plt.xlabel("C")
plt.colorbar()
plt.savefig("influence_C_gamma.png")