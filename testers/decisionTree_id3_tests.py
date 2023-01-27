import pandas as pd
from models.decision_tree_id3 import ID3Tree
from sklearn.metrics import classification_report

from preprocessing import cars

df = pd.read_csv('../datasets/fever.csv', delimiter=',')

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

#X,y = mushrooms.preprocess_dataset()
#X,y  = weather.preprocess_dataset()
X,y = cars.preprocess_dataset_cars()
#tree = ID3Tree(split_features_fun=None, fnames=['outlook','temperature','humidity','wind'],classnames=['yes','no'])
tree = ID3Tree(max_depth = 1000, min_samples_split = 2)
tree.fit(X,y)

y_pred = tree.predict(X)

print(classification_report(y, y_pred))

# # from sklearn import tree
# id3_pro = DecisionTreeID3(max_depth = 1000, min_samples_split = 2,min_gain=0.0)
# id3_pro.fit(pd.DataFrame(X),pd.DataFrame(y).iloc[:,0])
# y_pred2 = id3_pro.predict(pd.DataFrame(X))
#
# print(classification_report(y, y_pred2))
