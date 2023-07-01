# We implement Bayesian networks.
# For simplificity, all probability variables are Booleans.
#
# The networks are represented as Python dictionaries. Below is a sketch
# of the network for the burglary example from the lecture notes.
# It states, for example, that P(alarm | burglary, not earthquake) = 0.94.
# It follows that $P(not alarm | burglary, not earthquake) = 1-0.94.
#
# example_network = {
#     'Burglary': {'name': 'Burglary', 'parents': [], 'probabilities': {(): 0.001}},
#     'Earthquake': {'name': 'Earthquake', 'parents': [], 'probabilities': {(): 0.002}},
#     'Alarm': {
#         'name': 'Alarm',
#         'parents': ['Burglary', 'Earthquake'],
#         'probabilities': {
#             (True, True): 0.95,
#             (True, False): 0.94,
#             (False, True): 0.29,
#             (False, False): 0.001}
#         },
#     'JohnCalls': {'name': 'JohnCalls', 'parents': ['Alarm'], 'probabilities': {(True,): 0.9, (False,): 0.05}},
#     'MaryCalls': {'name': 'MaryCalls', 'parents': ['Alarm'], 'probabilities': {(True,): 0.7, (False,): 0.01}}
# }
#
# Queries consist of the network (as above), a single query variable, and an atomic event for the evidence.
# The latter is a dictionary that gives for the every evidence variable a value.
#
# Example query: query(example_network, 'Burglary', {'MaryCalls':True, 'JohnCalls':True})

def query(network, node, evidence):
    return 0.5    # TODO: compute actual value
    # need to compute P(node = true | evidence)
    # marginalize over all hidden variables (i.e., all variables in network except node (= query variable) and evidence)
    # yields big formula with lots of conditional probabilities
    # simplify the formula by throwing out conditions according to the network
    # fill in concrete values for the remaining conditional probabilities as given by the network
