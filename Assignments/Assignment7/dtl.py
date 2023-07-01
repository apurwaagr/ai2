import math

def dtl(examples, attributes, target, default):
    # Base cases
    if len(examples) == 0:
        return default
    elif all(examples[0][target] == example[target] for example in examples):
        return examples[0][target]
    elif len(attributes) == 0:
        return pluralityValue(examples, target, default)
    else:
        best_attr = chooseAttribute(attributes, examples, target)
        tree = (best_attr, {})
        attr_values = attributes[best_attr]
        remaining_attributes = attributes.copy()
        del remaining_attributes[best_attr]

        for value in attr_values:
            subset = getExamplesWithValue(examples, best_attr, value)
            if len(subset) == 0:  # Handle empty subset
                tree[1][value] = default
            else:
                subtree = dtl(subset, remaining_attributes, target, pluralityValue(examples, target, default))
                tree[1][value] = subtree

        return tree


def pluralityValue(examples, target, default):
    value_counts = {}
    for example in examples:
        value = example[target]
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_values[0][0]


def chooseAttribute(attributes, examples, target):
    best_attr = None
    best_gain = -1

    for attr in attributes:
        gain = informationGain(attr, examples, target)
        if gain > best_gain:
            best_attr = attr
            best_gain = gain

    return best_attr


def informationGain(attribute, examples, target):
    total_entropy = entropy(examples, target)
    attr_values = set([example[attribute] for example in examples])
    weighted_entropy = 0

    for value in attr_values:
        subset = getExamplesWithValue(examples, attribute, value)
        subset_entropy = entropy(subset, target)
        weighted_entropy += (len(subset) / len(examples)) * subset_entropy

    information_gain = total_entropy - weighted_entropy
    return information_gain


def entropy(examples, target):
    value_counts = {}
    for example in examples:
        value = example[target]
        if value in value_counts:
            value_counts[value] += 1
        else:
            value_counts[value] = 1

    entropy = 0
    total_examples = len(examples)
    for count in value_counts.values():
        probability = count / total_examples
        entropy -= probability * math.log2(probability)

    return entropy


def getExamplesWithValue(examples, attribute, value):
    return [example for example in examples if example[attribute] == value]



# Example usage
if __name__ == '__main__':
    # Example query
    tree = dtl(
        examples=[
            {'Furniture': 'No', 'Nr. of rooms': '3', 'New kitchen': 'Yes', 'Acceptable': 'Yes'},
            {'Furniture': 'Yes', 'Nr. of rooms': '3', 'New kitchen': 'No', 'Acceptable': 'No'},
            {'Furniture': 'No', 'Nr. of rooms': '4', 'New kitchen': 'No', 'Acceptable': 'Yes'},
            {'Furniture': 'No', 'Nr. of rooms': '3', 'New kitchen': 'No', 'Acceptable': 'No'},
            {'Furniture': 'Yes', 'Nr. of rooms': '4', 'New kitchen': 'No', 'Acceptable': 'Yes'}
        ],
        attributes={'Furniture': ['Yes', 'No'], 'Nr. of rooms': ['3', '4'], 'New kitchen': ['Yes', 'No'],
                    'Acceptable': ['Yes', 'No']},
        target='Acceptable',
        default='Yes'
    )

    print(tree)
