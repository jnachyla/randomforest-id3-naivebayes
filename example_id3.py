import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import mushrooms
from ID3Github import DecisionTreeID3
from tree_id3 import ID3Tree
from information_gain import information_gain
from information_gain import entropy

df = pd.read_csv('fever.csv', delimiter=',')

# # case 1
# tree = ID3Tree()
Sy = df[['infected']].values
df.drop(columns=['infected'], inplace=True)
Sx = df.values
#
# print(entropy(Sy))
# print(information_gain(Sx=Sx, Sy = Sy, a_idx=0, entropy_S=0.99))
#
# node = tree._build_tree([], Sx, Sy, None)
# assert node.predicted_class == 1

# tree = ID3Tree(split_features_fun=None, fnames=["fever", "cough","bi"], classnames=["notinfected","infected"])
# attr_idxs = np.array(list(range(len(Sx[0, :]))))
# tree.fit(X=Sx, y=Sy)
#tree.printTree()

from sklearn import datasets
iris = datasets.load_iris()

X,y = mushrooms.preprocess_dataset()
tree = ID3Tree(split_features_fun=None)
tree.fit(X,y)

y_pred = tree.predict(X)

print(classification_report(y, y_pred))

from sklearn import tree
id3_pro = DecisionTreeID3(max_depth = 1000, min_samples_split = 2,min_gain=0.0)
id3_pro.fit(pd.DataFrame(X),pd.DataFrame(y).iloc[:,0])
y_pred2 = id3_pro.predict(pd.DataFrame(X))

print(classification_report(y, y_pred2))
