'''
    Your assignment is to implement the following functions,
    i.e. replace the `raise NotImplementedError()` with code that returns the right answer.

    Do not use any libraries!

    We will partly automate the evaluation so please make sure you
    that you don't change the signature of functions.
    Also please use a recent python version (at least version 3.6).
    
    You may call functions that you've implemented in other functions.
    You may also implement helper functions.

    Understanding the arguments:
        Omega:  the sample space, represented as a list
        P:      the probability function (P : Omega ⟶ float)
        ValX:   the possible values of a random variable X, represented as a list
        VarX:   a random variable (VarX : Omega ⟶ ValX), here represented as a function
        x:      a concrete value for VarX (x ∈ ValX)
        EventA: an atomic event a, represent as list of pairs [(VarA1, a1), (VarA2, a2), ...]
                representing the event a := (VarA1=a1) ∧ (VarA2=a2) ∧ …

    Example code: given Omega, P, VarX, x
        w = Omega[0]       # pick the first sample (note that the order is meaningless)
        print(P(w))        # print the probability of that sample
        if VarX(w) == x:   # compute the value of the random variable for this sample and compare it to x
            print('X = x holds for this sample')
        else:
            print('X = x doesn't hold for this sample')

    Example call:
        def isEven(n):
            if n%2 == 0:
                return 'yes'
            else:
                return 'no'
        def probfunction(n):
            return 1/6    # fair die
        print('P(isEven = yes) for a fair die:')
        print(unconditional_probability([1,2,3,4,5,6], probfunction, isEven, 'yes'))
'''


def unconditional_probability(Omega, P, VarX, x):
    ''' P(VarX = x)
        Hint: Add up P(ω) for all those ω where VarX(ω) == x by iterating over the values ω in Omega
    '''
    # raise NotImplementedError()
    prob = 0.0
    for w in Omega:
        if VarX(w) == x:
            prob += P(w)
    return prob

def unconditional_joint_probability(Omega, P, EventA):
    ''' P(a) '''
    # raise NotImplementedError()
    prob = 0.0
    for w in Omega:
        okay = True
        for a in EventA:
            if a[0](w) != a[1]:
                okay = False
                break
        if okay:
            prob += P(w)
    return prob

def conditional_probability(Omega, P, VarX, x, VarY, y):
    ''' P(VarX=x|VarY=y) '''
    # raise NotImplementedError()
    return unconditional_joint_probability(Omega, P, [(VarX, x), (VarY, y)]) / unconditional_probability(Omega, P, VarY, y)

def conditional_joint_probability(Omega, P, EventA, EventB):
    ''' P(a|b) '''
    # raise NotImplementedError()
    return unconditional_joint_probability(Omega, P, EventA + EventB) / unconditional_joint_probability(Omega, P, EventB)   

def probability_distribution(Omega, P, VarX, ValX):
    ''' P(VarX),
        which is defined [P(VarX = x0), P(VarX = x1), …] where ValX = [x0, x1, …]
        (return a list)
    '''
    # raise NotImplementedError()
    return [unconditional_probability(Omega, P, VarX, val) for val in ValX]

def conditional_probability_distribution(Omega, P, VarX, ValX, VarY, ValY):
    ''' P(VarX|VarY)
        to be represented as a python dictionary of the form
        {(x0, y0) : P(VarX=x0|VarY=y0), …}
        for all pairs (x_i, y_j) ∈ ValX × ValY
    '''
    # raise NotImplementedError()
    return {(x,y) : conditional_probability(Omega, P, VarX, x, VarY, y)
            for x in ValX for y in ValY}

def test_event_independence(Omega, P, EventA, EventB):
    ''' P(a,b) = P(a) ⋅ P(b)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    # raise NotImplementedError()
    pab = unconditional_joint_probability(Omega, P, EventA + EventB)
    pa = unconditional_joint_probability(Omega, P, EventA)
    pb = unconditional_joint_probability(Omega, P, EventB)
    return pab - 1e-12 < pa*pb < pab + 1e-12

def test_variable_independence(Omega, P, VarX, ValX, VarY, ValY):
    ''' P(X,Y) = P(X) ⋅ P(Y)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    # raise NotImplementedError()
    for x in ValX:
        for y in ValY:
            if not test_event_independence(Omega, P, [(VarX, x)], [(VarY, y)]):
                return False
    return True

def test_conditional_independence(Omega, P, EventA, EventB, EventC):
    ''' P(a,b|c) = P(a|c) ⋅ P(b|c)
        (return a bool)
        Note: Due to rounding errors, you should only test for approximate equality (the details are up to you)
    '''
    # raise NotImplementedError()
    pabc = conditional_joint_probability(Omega, P, EventA + EventB, EventC)
    pac = conditional_joint_probability(Omega, P, EventA, EventC)
    pbc = conditional_joint_probability(Omega, P, EventB, EventC)
    return pabc - 1e-12 < pac*pbc < pabc + 1e-12

