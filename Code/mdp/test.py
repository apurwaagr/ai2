import numpy as np
import mdp_solved as mdp
import sys


# TEST SETUP

wall = { 'iswall': True, 'isterminal': False, 'reward': None }
t = lambda r: { 'iswall': False, 'isterminal': True, 'reward': r }       # terminal state
n = lambda r: { 'iswall': False, 'isterminal': False, 'reward': r }      # normal state

TEST_WORLD = [
    [n(-4), n(-1), n(-1), n(-1), n(-1),  n(-1), n(-1)],
    [n(-2), n(-1), n(-1), n(-1), n(-1),  n(-1), t(10)],
    [n(-1), n(-1), n(-3), n(-4), n(-85), n(-1), n(-1)],
    [n(-1), wall,  n(-3), n(-3), n(-1),  n(-1), n(-1)],
    [n(-1), n(-1), wall,  n(-3), n(-4),  n(-1), n(-1)],
    [n(-1), n(-1), n(-1), n(-4), n(-3),  n(-1), t(5)],
]

WORLD = np.array(TEST_WORLD).transpose()[:,::-1]


P = 0.7
EPSILON = 0.0000000001
GAMMA = 0.8


# CALL VALUE ITER FUNCTION

u, p = mdp.valueiter(WORLD, GAMMA, EPSILON, P)


# EXPECTED RESULTS

u_target = np.array([[-4.82679219e+00, -4.75649682e+00, -4.65446810e+00, -4.53106385e+00, -5.29794744e+00, -7.10926040e+00],
       [-4.86615760e+00, -4.82679219e+00,  0.00000000e+00, -4.17281068e+00, -3.39483667e+00, -2.89356331e+00],
       [-4.90137929e+00,  0.00000000e+00, -8.26640884e+00, -6.06645408e+00, -2.76227105e+00, -2.03384878e+00],
       [-7.24380238e+00, -7.28646126e+00, -7.31021245e+00, -8.49037511e+00, -1.79946110e+00, -8.18418570e-01],
       [-2.67887633e+00, -4.53567764e+00, -4.31605730e+00, -8.39099883e+01, -8.06181179e-02,  8.85226769e-01],
       [ 2.11941097e+00,  5.42347137e-01,  8.42897296e-01, 2.88859418e+00,  5.32991853e+00,  3.19406023e+00],
       [ 5.00000000e+00,  2.11941097e+00,  2.55568632e+00, 5.62117193e+00,  1.00000000e+01,  5.66282640e+00]])

p_target = np.array([['up', 'up', 'up', 'right', 'right', 'right'],
       ['left', 'left', '', 'up', 'right', 'right'],
       ['left', '', 'up', 'up', 'up', 'right'],
       ['right', 'right', 'right', 'left', 'up', 'right'],
       ['right', 'right', 'down', 'right', 'up', 'right'],
       ['right', 'right', 'right', 'right', 'right', 'right'],
       ['', 'down', 'up', 'up', '', 'down']], dtype='<U5')

# Compute naive solution (try going to neighbour with maximum utility)
p_naive = np.zeros(p_target.shape, dtype=p_target.dtype)
for x in range(p_target.shape[0]):
    for y in range(p_target.shape[1]):
        if WORLD[x,y]['iswall'] or WORLD[x,y]['isterminal']:
            continue
        best = np.min(u_target) - 1
        for (v, x2, y2) in [('left', x-1, y), ('right', x+1, y), ('up', x, y+1), ('down', x, y-1)]:
            if x2 < 0 or x2 >= u_target.shape[0] or y2 < 0 or y2 >= u_target.shape[1]:
                continue
            if WORLD[x2,y2]['iswall'] or WORLD[x2,y2]['isterminal']:
                continue
            if u_target[x2, y2] > best:
                best = u_target[x2, y2]
                p_naive[x, y] = v


# COMPARE RESULTS

relevant = (p_target[:,:] != '')
if np.max(np.abs(u_target[relevant]-u[relevant])) > 1e-5:
    print('Computed utilities are wrong!')
    sys.exit(1)


if not (p[relevant] == p_target[relevant]).all():
    print('Computed policy is wrong!')
    if (p[relevant] == p_naive[relevant]).all():
        print('The naive policy (go to direction of neighbour with maximal utility) has been used')
    sys.exit(1)

print('Everything is okay')
