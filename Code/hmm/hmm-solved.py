'''
TODO: implement the functions hmm_* below and upload this file (with the name `hmm.py`).
Parameters:
    X0: a numpy array with the distribution at time 0.   (X0[i] = P(X_0=i))
    T:  a 2-d numpy array representing the transition, following the conventions in the lecture.
    O:  the sensor model with O_ij = P(e=i | X=j) represented as a 2-d numpy array.
        From this, the diagonal matrices O_t can be computed using the evidence at time t.
    e:  the evidence as a python list. Following the conventions of the lecture, evidence starts at t=1, i.e. e[i] corresponds to t=i+1.
    k:  the extra time index needed for prediction and smoothing.

You may use `test.py` for testing your solution.
'''


import numpy as np    # don't use any other libraries


def hmm_filter(X0, T, O, e):
    # return X0
    f = X0
    for ev in e:
        f = np.dot(np.diag(O[ev]), np.dot(T.transpose(), f))
        f /= np.sum(f)  # normalization
    return f

def hmm_predict(X0, T, O, e, k):
    # return X0
    assert k > len(e)
    f = hmm_filter(X0, T, O, e)
    for _ in range(k - len(e)):
        f = np.dot(T.transpose(), f)
    return f

def hmm_smooth(X0, T, O, e, k):
    # return X0
    assert k < len(e)
    f = hmm_filter(X0, T, O, e[:k])
    b = np.ones(len(X0))
    for ev in e[:k-1:-1]:
        b = np.dot(T, np.dot(np.diag(O[ev]), b))
    return f*b / np.sum(f*b)    # normalization

