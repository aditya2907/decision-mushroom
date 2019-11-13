"""
Get and process dataset on mushrooms consisting of labels ('p (poisonous) vs.
e(edible)), and 22 nominal features. Fix missing values based on plurality.
Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""

from collections import Counter


def missing_values(example, funct):
    """
    Tell what features are missing a value for an example
    :param example: a list of feature values of an example
    :param funct: function for determining what counts as a missing value
    :return: indices of features with a missing value
    """
    return [i for i in range(len(example)) if funct(example[i])]


def fix_missing_values(examples):
    """
    Fix missing feature values by filling in with the value that is most
    common for the feature.
    :param examples: list of examples (labels + features)
    """
    # Get the feature values that are most common in each column
    column_pluralities = dict()

    for col in range(len(examples[0])):
        column_values = [row[col] for row in examples]
        column_counter = Counter(column_values)
        column_pluralities[col] = column_counter.most_common()[0]

    # Fill in missing features with the most common value in the column
    for ex in examples:
        missing_vals = missing_values(ex, lambda x: '?' in x)

        for i in missing_vals:
            ex[i] = column_pluralities[i][0]


def convert_data(examples):
    """
    Convert string labels and feature values of examples to integers.
    :param examples: list of labels and feature values
    """
    # Convert column by column
    for col in range(len(examples[0])):
        # Assign each value of an feature to an integer
        value_set = list(set([row[col] for row in examples]))

        # Replace string data in the column with integers
        for row in examples:
            row[col] = value_set.index(row[col])


def get_data():
    """
    Get and process mushroom dataset, fix missing values
    :return: a list of mushroom features and a list of mushroom labels
    """
    # Get data from file
    with open('agaricus-lepiota.data', 'r') as f:
        file_lines = f.read().split('\n')

    # Remove punctuation
    examples = [l.split(',') for l in file_lines if l]
    # Fill in missing values
    fix_missing_values(examples)
    # Convert values to integers
    convert_data(examples)
    # Separate features and labels
    examples = [(e[1:], e[0]) for e in examples]

    return examples
