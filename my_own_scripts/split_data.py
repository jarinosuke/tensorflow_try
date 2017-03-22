import numpy as np
import pandas as pd # to load csv super easily

df = pd.read_csv("../dataset/voice.csv", header=0)

labels = (df["label"] == "male").values * 1
labels = labels.reshape(-1, 1)

del df["label"]
data = df.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                               test_size=0.3, random_state=123456)

tmp_dir = "../tmp/"
np.save(tmp_dir + "X_train.npy", X_train) 
np.save(tmp_dir + "X_test.npy", X_test)
np.save(tmp_dir + "y_train.npy", y_train)
np.save(tmp_dir + "y_test.npy", y_test)
