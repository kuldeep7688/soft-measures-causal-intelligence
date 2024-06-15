"""
Constants for use throughout the project.
"""

# Special tokens used in entity recognition tasks
E1_START_TOKEN = '<e1>'
E1_END_TOKEN = '</e1>'
E2_START_TOKEN = '<e2>'
E2_END_TOKEN = '</e2>'

# Mapping of relationship labels to unique identifier numbers
# This is used for converting relationship labels into numerical form for model training and evaluation
LABEL_TO_ID = {
    'no_relation': 0,
    'headquartered_in': 1,
    'formed_in': 2,
    'title': 3,
    'shares_of': 4,
    'loss_of': 5,
    'acquired_on': 6,
    'agreement_with': 7,
    'operations_in': 8,
    'subsidiary_of': 9,
    'employee_of': 10,
    'attended': 11,
    'cost_of': 12,
    'acquired_by': 13,
    'member_of': 14,
    'profit_of': 15,
    'revenue_of': 16,
    'founder_of': 17,
    'formed_on': 18,
}
