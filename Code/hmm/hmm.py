'''
Task: implement the functions hmm_* below and upload this file (with the name `hmm.py`).
Parameters:
    X0: a numpy array with the distribution at time 0.   (X0[i] = P(X_0=i))
    T:  a 2-d numpy array representing the transition model, following the conventions in the lecture.
    O:  the sensor model with O_ij = P(E=i | X=j) represented as a 2-d numpy array.
        From this, the diagonal matrices O_t can be obtained using the evidence at time t.
    e:  the observed evidence e_1,... as a python list. Following the conventions of the lecture, evidence starts at t=1, i.e. e[i] corresponds to t=i+1.
    k:  the extra time index needed for prediction and smoothing.

You may use `test.py` for testing your solution.
'''


import numpy as np    # don't use any other libraries


def hmm_filter(X0, T, O, e):
    ''' Computes P(X_t | e_{1:t}) where t := len(e) '''
    return X0   # TODO

def hmm_predict(X0, T, O, e, k):
    ''' Computes P(X_k | e_{1:t}) where t := len(e) and k > t.
        Note that `k` is an absolute time index here while the lecture notes use `k` as a relative offset. '''
    return X0   # TODO

def hmm_smooth(X0, T, O, e, k):
    ''' Computes P(X_k | e_{1:t}) where t := len(e) and k < t. '''
    return X0   # TODO

