'''
    Please implement the function `valueiter` and submit the file with the name `mdp.py`.
    You are welcome (and encouraged) to experiment with different worlds and parameters.

    You don't need any libraries other than numpy (and tkinter for the visualization).
    Aside from those, you may not use any libraries that are not part of the standard library.
'''

import numpy as np


def valueiter(world, gamma, epsilon, p):
    '''
        The value iteration algorithm.
        Parameters:
            world: a numpy array describing the world.
                Each element is a dictionary with three keys:
                    'iswall':     is True iff the square is a wall
                    'isterminal': is True iff the square/state is terminal
                    'reward':     the reward of the square/state
            gamma: the discount factor γ
            epsilon: the maximum error ε
            p: the probability of going into the right direction (see assignment)
        Returns: a pair (utilities, policy) represented as numpy arrays (see example code below)
    '''

    width, height = world.shape

    # initialize an array for the utilities (everything 0)
    utilities = np.zeros((width, height))

    # initialize an array for the policy
    # it should be filled with the strings 'left', 'right', 'up' or 'down'
    policy = np.zeros(world.shape, dtype='<U5')
    policy[:,:] = 'right'   # for now, the policy is: always go right

    # We initialize the utilities with the rewards.
    for x in range(width):
        for y in range(height):
            if world[x,y]['iswall']:   # we don't care about walls
                continue
            utilities[x,y] = world[x,y]['reward']

    # TODO
    # Returning the initial values here, the visualization already works.
    # But of course, we have to do the actual iteration here before returning.
    
    return utilities, policy


def visualize(world, utilities, policy):
    '''
        This function opens a window to visualize the rewards, utilities and policy.
        You do not have to understand its implementation.
    '''
    import tkinter
    # normalize to [0,1]
    normalizedutilities = (utilities - np.min(utilities)) / (np.max(utilities) - np.min(utilities) + 0.000001)
    def utilitytocolor(u):
        c = [(100, 0, 0), (150, 150, 0), (0, 200, 0)]
        l = len(c)-1
        i = int(u*l)
        f = u*l-i
        r = c[i][0]*(1-f)+c[i+1][0]*f
        g = c[i][1]*(1-f)+c[i+1][1]*f
        b = c[i][2]*(1-f)+c[i+1][2]*f
        hex2 = lambda i : hex(i)[2:].ljust(2,'0')
        return f'#{hex2(int(r))}{hex2(int(g))}{hex2(int(b))}'

    window = tkinter.Tk()
    window.title('MDP')
    width, height = world.shape
    for x in range(width):
        for y in range(height):
            if world[x,y]['iswall']:
                bgcolor = '#888888'
            else:
                bgcolor = '#ffffff'
            c = tkinter.Canvas(window, width=60, height=60, bg=bgcolor)
            if not (world[x,y]['iswall'] or world[x,y]['isterminal']):
                c.create_text(30, 37, anchor='c', text={'left':'←','right':'→','up':'↑','down':'↓'}[policy[x,y]])
            if world[x,y]['iswall']:
                c.create_text(30, 37, anchor='c', text='W')
            if world[x,y]['isterminal']:
                c.create_text(30, 37, anchor='c', text='T')
            if not world[x,y]['iswall']:
                fillcolor = utilitytocolor(normalizedutilities[x,y])
                c.create_text(30, 50, anchor='c', text=f'{utilities[x,y]:.3f}', fill=fillcolor)
                c.create_text(30, 23, anchor='c', text=f'{world[x,y]["reward"]:.3f}', fill='#888888')
            c.create_text(30, 9, anchor='c', text=f'{(x,y)}')
            c.grid(column=x, row=height-y)
    window.mainloop()


# EXAMPLE WORLDS
# Note: the squares will be re-arranged to match the expected indices (e.g. (0,0) for bottom left)

# helpers for creating states
wall = { 'iswall': True, 'isterminal': False, 'reward': None }
t = lambda r: { 'iswall': False, 'isterminal': True, 'reward': r }       # terminal state
n = lambda r: { 'iswall': False, 'isterminal': False, 'reward': r }      # normal state

# World from the lecture slides (4x3 world)
WORLD_SLIDES = [
    [n(-0.04), n(-0.04), n(-0.04), t(1)    ],
    [n(-0.04), wall,     n(-0.04), t(-1)   ],
    [n(-0.04), n(-0.04), n(-0.04), n(-0.04)]
]

# World from the presence problem
WORLD_PRESENCE = [
    [n(-1), n(-1), t(10)],
    [n(-1), n(-1), n(-1)],
    [n(-1), n(-1), n(-1)],
]

# A bigger world
WORLD_BIG = [
    [n(-1), n(-1), n(-1), n(-1), n(-1),  n(-1), n(-1)],
    [n(-1), n(-1), n(-1), n(-1), n(-1),  n(-1), t(10)],
    [n(-1), n(-1), n(-3), n(-4), n(-85), n(-1), n(-1)],
    [n(-1), wall,  n(-3), n(-3), n(-1),  n(-1), n(-1)],
    [n(-1), n(-1), wall,  n(-3), n(-4),  n(-1), n(-1)],
    [n(-1), n(-1), n(-1), n(-4), n(-3),  n(-1), t(5)],
]

if __name__ == '__main__':
    # CHANGE THESE TO TRY OUT DIFFERENT PROBLEMS
    WORLD = WORLD_SLIDES
    P = 0.8                 # probability to go in desired direction
    EPSILON = 0.001
    GAMMA = 0.95

    # run the algorithm
    world = np.array(WORLD).transpose()[:,::-1]
    utilities, policy = valueiter(world, GAMMA, EPSILON, P)
    visualize(world, utilities, policy)

