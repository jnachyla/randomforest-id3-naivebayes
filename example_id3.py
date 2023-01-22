import numpy as np
import pandas as pd
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

tree = ID3Tree(split_features_fun=None, fnames=["fever", "cough","bi"])
attr_idxs = np.array(list(range(len(Sx[0, :]))))
tree.fit(X=Sx, y=Sy)
tree.printTree()