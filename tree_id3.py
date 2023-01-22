import math
from collections import deque

from information_gain import information_gain
from information_gain import entropy
import numpy as np

class Node(object):
    def __init__(self):

        self.predicted_class = None
        # for real attributes
        self.thresholds = []
        self.value = None
        self.feature_idx = None
        self.children = []
        self.fname = None


class ID3Tree(object):
    def __init__(self, split_features_fun = None, fnames = None):
        '''

        :param split_features_fun: możliwe wartości None, log2, sqrt

        '''
        self.fnames = fnames
        self.split_features_fun = split_features_fun
        self.root = None

    def fit(self, X, y):

        A_ids = np.array(list(range(len(X[0, :]))))

        self._build_tree(A_ids, X, y, None)



    def _build_tree(self, A_ids:np.ndarray, Sx:np.ndarray, Sy:np.ndarray, node:Node):
        if not node:
            node = Node()
            self.root = node

        # warunki zakończenia
        #czy wszystkie Sy maja ta sama klase
        uniq_classes_freqs = np.unique(Sy, return_counts=True)
        if len(uniq_classes_freqs[0]) == 1:
            node.predicted_class = uniq_classes_freqs[0]
            return node

        # if A jest pusty then
        #     return stwórz liść(label=najczęstsza wartość ytarget)
        if len(A_ids) == 0:
            most_freq_class_idx = np.argmax(uniq_classes_freqs[1])
            node.predicted_class = uniq_classes_freqs[0][most_freq_class_idx]
            return node

        A_to_split = np.array(A_ids)
        if  self.split_features_fun:
            if self.split_features_fun == "log2":
                A_to_split = np.random.choice(A_ids, int(math.log2(len(A_ids))))
            elif self.split_features_fun == "sqrt":
                A_to_split = np.random.choice(A_ids, int(math.sqrt(len(A_ids))))
            else:
                raise Exception("Unkonwn value of  split_features_fun", self.split_features_fun)

        entropy_S = entropy(Sy)
        igs_A_sampled = [information_gain(Sx=Sx, Sy=Sy, a_idx=attr_idx, entropy_S=entropy_S) for attr_idx in A_to_split]

        best_attr = A_to_split[np.argmax(igs_A_sampled)]
        if self.fnames:
           node.fname = self.fnames[best_attr]

        node.feature_idx = best_attr

        Sx_a_best = Sx[:, best_attr]
        best_attr_values = np.unique(Sx_a_best, return_counts=False)

        # jeśli wartości jest więcej niż jedna to bierzemy środkową wartość

        best_attr_values = np.sort(best_attr_values)

        # FIXME do wywalenia
        # if best_attr_values > 2:
        #     # threshold jest traktowany priorytetowo dla atrybutów rzeczywistych stosujemy binarny podział
        #     node.thresholds = np.array([np.mean(best_attr_values)])
        #     best_attr_values = [0,1]

        for v in best_attr_values:
            child_node = Node()
            child_node.value = v
            node.children.append(child_node)

            # partycjonowanie
            value_mask = self.create_mask(Sx, v, best_attr)

            if len(value_mask) == 0:
                most_freq_class_idx = np.argmax(uniq_classes_freqs[1])
                child_node.predicted_class = uniq_classes_freqs[0][most_freq_class_idx]
            else:
                Sx_new = Sx[value_mask]
                Sy_new = Sy[value_mask]
                A_ids = np.delete(A_ids, np.where(A_ids == best_attr))
                self._build_tree(Sx=Sx_new, Sy=Sy_new, A_ids = A_ids, node = child_node)



    def create_mask(self, X, val, attr_id):
        return [i for i, x in enumerate(X[:,attr_id]) if x == val]


    def predict(self, X):
        return

    def printTree(self):
        if not self.root:
            return
        nodes = deque()
        nodes.append(self.root)
        while len(nodes) > 0:
            node = nodes.popleft()
            print('({}{})'.format(node.value, node.fname))
            if node.children:
                for child in node.children:
                    print('({})'.format(child.value))
                    nodes.append(child)






