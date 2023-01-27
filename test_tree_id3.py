from unittest import TestCase

import numpy as np

from tree_cart import Node
from tree_id3 import ID3Tree


class TestID3Tree(TestCase):
    def test_predict(self):
        root = Node()

        n1 = Node()
        n1.value = 2
        n1.feature_idx = 1

        n11 = Node()
        n11.predicted_class = 0

        n1.children = [n11]

        n2 = Node()
        n2.value = 3
        n2.feature_idx = 2

        n22 = Node()
        n22.predicted_class = 1
        n2.children = [n22]

        root.children = [n1, n2]

        alg = ID3Tree()

        X1 = np.array([10, 2, 4])
        X2 = np.array([10, 3, 3])
        assert alg.predict(root, X1) == 0
        assert alg.predict(root, X2) == 1
