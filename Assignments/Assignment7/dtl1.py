
# Example query:
# dtl(
#    examples = [
#         {'Furniture': 'No', 'Nr. of rooms': '3', 'New kitchen': 'Yes', 'Acceptable': 'Yes'},
#         {'Furniture': 'Yes', 'Nr. of rooms': '3', 'New kitchen': 'No', 'Acceptable': 'No'},
#         {'Furniture': 'No', 'Nr. of rooms': '4', 'New kitchen': 'No', 'Acceptable': 'Yes'},
#         {'Furniture': 'No', 'Nr. of rooms': '3', 'New kitchen': 'No', 'Acceptable': 'No'},
#         {'Furniture': 'Yes', 'Nr. of rooms': '4', 'New kitchen': 'No', 'Acceptable': 'Yes'}
#    ],
#    attributes = {'Furniture': ['Yes', 'No'], 'Nr. of rooms': ['3', '4'], 'New kitchen': ['Yes', 'No'], 'Acceptable': ['Yes', 'No']},
#    target = 'Acceptable',
#    default = 'Yes'
# )
#
# Warning: the target attribute must not be used in the decision tree
# Warning: attributes are not necessarily binary
#
#
# Expected result:
# ('Nr. of rooms', {
#     '4': 'Yes',
#     '3': ('New kitchen', {
#         'Yes': 'Yes',
#         'No': 'No'}
#     )
#     }
# )



def dtl(examples, attributes, target, default):
    return default
