from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

mnist = loadmat("./mnist-original.mat")
mnist_data_training = np.array(mnist["data"].T[:60000], dtype=np.int16)
mnist_label_training = np.array(mnist["label"][0][:60000], dtype=np.int16)
mnist_data_test = mnist["data"].T[60000:]
mnist_label_test = mnist["label"][0][60000:]


def distance(train_index: int, test_index: int):
    m = np.array(
        (mnist_data_training[train_index] - mnist_data_test[test_index]), dtype=np.int64)
    return np.sum(m**4)


def KNN(k: int, test_index: int):
    distances = [distance(i, test_index)
                 for i in range(len(mnist_data_training))]
    idx = np.argpartition(distances, k)
    clas, freq = np.unique(mnist_label_training[idx[:k]], return_counts=True)
    return clas[np.argmax(freq)]


pred = np.empty(len(mnist_data_test), dtype=np.int64)
for i in range(0, 10000, 100):
    pred[i] = KNN(1, i)
    print(f"Progress: {i/100}")

# print(pred[:10])
# print(mnist_label_test[:10])
correct = np.zeros(len(mnist_data_test), dtype=bool)
for i in range(0, 10000, 100):
    if (pred[i] == mnist_label_test[i]):
        correct[i] = 1
# print(correct)
print(sum(correct), '%')
