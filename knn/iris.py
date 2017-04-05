from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.cross_validation import KFold
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


dataset = load_iris()

features = dataset.data
featur_names = list(dataset.feature_names)
target = dataset.target
target_names = list(dataset.target_names)

markers = ['bo', 'rs', 'g^']

for t in range(3):
    plt.plot(features[target == t, 2], features[target == t, 0], markers[t], label=target_names[t])
plt.grid(True)
plt.legend()
plt.show()

classifier = KNeighborsClassifier(n_neighbors=1)
mean = []

for training, testing in KFold(len(features), n_folds=5, shuffle=True):
    classifier.fit(features[training], target[training])
    prediction = classifier.predict(features[testing])
    mean.append(np.mean(prediction == target[testing]))

print("Accuracy at every step: %s" % ', '.join(['%s : %s' % (str(mean.index(i)), i) for i in mean]))
print("Average accuracy = %f" % np.mean(mean))
print("Median accuracy = %f" % np.median(mean))
