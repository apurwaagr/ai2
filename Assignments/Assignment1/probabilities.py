def unconditional_probability(Omega, P, VarX, x):
    """
    Returns the unconditional probability of VarX == x.
    """
    count = 0
    total = 0
    for w in Omega:
        if VarX(w) == x:
            count += P(w)
        total += P(w)
    return count / total

def unconditional_joint_probability(Omega, P, EventA):
    """
    Returns the joint probability of the events in EventA.
    """
    total = 0
    for w in Omega:
        matches_event = True
        for VarA, a in EventA:
            if VarA(w) != a:
                matches_event = False
                break
        if matches_event:
            total += P(w)
    return total

def conditional_probability(Omega, P, VarX, x, VarY, y):
    """
    Returns the conditional probability of VarX == x given VarY == y.
    """
    count_XY = 0
    count_Y = 0
    for w in Omega:
        if VarY(w) == y:
            count_Y += P(w)
            if VarX(w) == x:
                count_XY += P(w)
    return count_XY / count_Y

def conditional_joint_probability(Omega, P, EventA, EventB):
    """
    Returns the joint conditional probability of EventA given EventB.
    """
    total_B = unconditional_joint_probability(Omega, P, EventB)
    return unconditional_joint_probability(Omega, P, EventA + EventB) / total_B

def probability_distribution(Omega, P, VarX, ValX):
    """
    Returns the probability distribution of VarX over the values in ValX.
    """
    dist = []
    for x in ValX:
        prob = 0
        for w in Omega:
            if VarX(w) == x:
                prob += P(w)
        dist.append(prob)
    return dist

def conditional_probability_distribution(Omega, P, VarX, ValX, VarY, ValY):
    """
    Returns the conditional probability distribution of VarX over the values in ValX, given VarY over the values in ValY.
    """
    dist = {}
    for x in ValX:
        for y in ValY:
            dist[(x, y)] = conditional_probability(Omega, P, VarX, x, VarY, y)
    return dist

def test_event_independence(Omega, P, EventA, EventB):
    """
    Tests whether the events in EventA and EventB are independent.
    """
    prob_AB = unconditional_joint_probability(Omega, P, EventA + EventB)
    prob_A = unconditional_joint_probability(Omega, P, EventA)
    prob_B = unconditional_joint_probability(Omega, P, EventB)
    return abs(prob_AB - prob_A * prob_B) < 1e-6

def test_variable_independence(Omega, P, VarX, ValX, VarY, ValY):
    """
    Tests whether VarX and VarY are independent.
    """
    prob_XY = unconditional_joint_probability(Omega, P, [((VarX, x), (VarY, y)) for x in ValX for y in ValY])

    prob_X = unconditional_probability(Omega, P, VarX, ValX)
    prob_Y = unconditional_probability(Omega, P, VarY, ValY)
    return abs(prob_XY - prob_X * prob_Y) < 1e-6

def test_conditional_independence(Omega, P, EventA, EventB, EventC):
    prob_ABC = conditional_joint_probability(Omega, P, EventA + EventB, EventC)
    prob_AC = conditional_joint_probability(Omega, P, EventA, EventC)
    prob_BC = conditional_joint_probability(Omega, P, EventB, EventC)
    return abs(prob_ABC - prob_AC * prob_BC) < 1e-6
