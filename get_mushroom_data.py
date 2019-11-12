"""
Get and process dataset on mushrooms consisting of labels ('p (poisonous) vs.
e(edible)), and 22 nominal attributes. Fix missing values based on plurality.
Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""

from collections import Counter


def missing_values(example, funct):
    """
    Tell what attributes are missing a value for an example
    :param example: a list of attribute values of an example
    :param funct: function for determining what counts as a missing value
    :return: indices of attributes with a missing value
    """
    return [i for i in range(len(example)) if funct(example[i])]


def fix_missing_values(examples):
    """
    Fix missing attribute values by filling in with the value that is most
    common for the attribute.
    :param examples: list of examples (labels + attributes)
    """
    # Get the attribute values that are most common in each column
    column_pluralities = dict()

    for col in range(len(examples[0])):
        column_values = [row[col] for row in examples]
        column_counter = Counter(column_values)
        column_pluralities[col] = column_counter.most_common()[0]

    # Fill in missing attributes with the most common value in the column
    for ex in examples:
        missing_vals = missing_values(ex, lambda x: '?' in x)

        for i in missing_vals:
            ex[i] = column_pluralities[i][0]


def get_data():
    """
    Get and process mushroom dataset, fix missing values
    :return: a list of mushroom attributes and a list of mushroom labels
    """
    # Get data from file
    with open('agaricus-lepiota.data', 'r') as f:
        file_lines = f.read().split('\n')

    # Remove punctuation
    examples = [l.split(',') for l in file_lines if l]
    # Fill in missing values
    fix_missing_values(examples)
    # Separate attributes and labels
    attributes = [e[1:] for e in examples]
    labels = [e[0] for e in examples]

    return attributes, labels
