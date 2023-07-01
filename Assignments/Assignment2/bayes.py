from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def query(network_dict, node, evidence):
    model = BayesianNetwork()
    for n in network_dict:
        model.add_node(n)
        parents = network_dict[n]['parents']
        if parents:
            model.add_edges_from([(p, n) for p in parents])
        cpd = TabularCPD(n, len(network_dict[n]['values']), network_dict[n]['probabilities'], parents)
        model.add_cpds(cpd)
    model.check_model()
    inference = VariableElimination(model)
    result = inference.query([node], evidence=evidence)
    return result[node].values[1]


example_network = {
    'Burglary': {
        'parents': [],
        'probabilities': [0.001, 0.999],
        'values': [True, False],
    },
    'Earthquake': {
        'parents': [],
        'probabilities': [0.002, 0.998],
        'values': [True, False],
    },
    'Alarm': {
        'parents': ['Burglary', 'Earthquake'],
        'probabilities': [
            0.95,
            0.94,
            0.29,
            0.001,
            0.05,
            0.06,
            0.71,
            0.999,
        ],
        'values': [True, False],
    },
    'JohnCalls': {
        'parents': ['Alarm'],
        'probabilities': [0.90, 0.10],
        'values': [True, False],
    },
    'MaryCalls': {
        'parents': ['Alarm'],
        'probabilities': [0.70, 0.30],
        'values': [True, False],
    },
}

result = query(example_network, 'Burglary', {'MaryCalls': True, 'JohnCalls': True})
print(result)
