"""
Train a decision tree on a dataset on mushrooms. Test the
decision tree in predicting whether a mushroom is poisonous or edible.
Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
from collections import Counter
import get_mushroom_data


def plurality(examples):
    counts = Counter([e[0] for e in examples])
    return counts.most_common()[0]


class Question:
    def __init__(self, funct, col_index):
        self.funct = funct
        self.col_index = col_index

    def answer(self, example):
        return self.funct(example[self.col_index])


class Tree_node:
    def __init__(self, true_branch, false_branch, question):
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.question = question


class Leaf():
    def __init__(self, examples):
        self.classification = plurality(examples)


def main():
    # Get mushroom data
    features, labels = get_mushroom_data.get_data()

    # Split data into training and test sets
    split = int(len(features)*0.8)
    train_features = features[:split]
    train_labels = labels[:split]
    test_features = features[split:]
    test_labels = labels[split:]


if __name__ == '__main__':
    main()