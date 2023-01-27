import math

import numpy as np


def entropy(Sy):
    Sy_counts = np.unique(Sy, return_counts=True)[1]

    return sum(-count / len(Sy) * math.log2(count / len(Sy)) for count in Sy_counts)

def entropy_counts(Sy, Sy_counts):
    return sum(-count / len(Sy) * math.log2(count / len(Sy)) for count in Sy_counts)



def information_gain(Sx, Sy, a_idx, entropy_S):
    '''
    :param Sx: X podzbiór X
    :param Sy: podzbiór y
    :param a_idx: indeks atrybutu
    :param entropy_S: Entropia dla S
    :return: Ingormation Gain atrybutu a_idx
    '''

    Sy_counts = np.unique(Sy, return_counts=True)[1]

    Sx_a = Sx[:, a_idx]
    S_a_counts = np.unique(Sx_a, return_counts=True)
    feature_vals = S_a_counts[0]

    feature_vals_freqs = S_a_counts[1] / len(Sx_a)

    feature_vals_mask = [
        [i
         for i, x in enumerate(Sx_a)
         if x == y]
        for y in feature_vals
    ]
    result = 0
    for mask, freq in zip(feature_vals_mask, feature_vals_freqs):
        result -= entropy(Sy[mask]) * freq

    return entropy_S + result





