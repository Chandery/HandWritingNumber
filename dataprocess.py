import numpy as np
import pandas as pd


data = pd.read_csv('archive/mnist_test.csv')

labels = np.array(data.iloc[:, 0])

imgs = np.array(data.iloc[:, 1:])

imgs = imgs.reshape((imgs.shape[0], 28, 28))

imgs = imgs[:,None,:,:]

print("labels_shape=",labels.shape)
print("imgs_shape=",imgs.shape)

print(np.isnan(labels).sum())
print(np.isnan(imgs).sum())

np.save('archive/test_labels.npy', labels)
np.save('archive/test_imgs.npy', imgs)

