"""
Train a decision tree on a dataset on mushrooms. Test the
decision tree in predicting whether a mushroom is poisonous or edible.
Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
from collections import Counter
import get_mushroom_data


class TreeNode:
    """
    A tree node consists of a 'true' and a 'false' branch (which could be
    Tree_node themselves or Leafs), and a Question by which the bucket is split
    into the two branches.
    """
    def __init__(self,  question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class Leaf:
    """
    A Leaf is a part of the decision tree that does not have branches. It is
    used to classify a label of an example.
    """
    def __init__(self, examples):
        self.classification = plurality(examples)


class Question:
    """
    A Question consists of a boolean function that takes a feature value, and
    the column index of the relevant feature in an example.
    """
    def __init__(self, funct, col_index):
        self.funct = funct
        self.col_index = col_index

    def answer(self, example):
        """
        Answer the Question based on the features of the example passed.
        :param example: a pair of feature list and label
        :return: True if the example's feature value fits the question,
        else False.
        """
        return self.funct(example[0][self.col_index])


def plurality(examples):
    """
    Return the most common label in a bucket.
    :param examples: list of examples in the bucket.
    :return: most common label in the bucket (nominal value).
    """
    counts = Counter([e[1] for e in examples])
    return counts.most_common()[0]


def is_homogeneous(examples):
    """
    Tell if the current bucket has only one type of label.
    :param examples: list of examples in the bucket.
    :return: True if there is only one type of label in the bucket, else False
    """
    return len(set([e[1]] for e in examples)) == 1


def gini_impurity(examples):
    return 0

def find_split_question(examples):
    """
    Find the best question to split a bucket with.
    :param examples: list of examples in the bucket.
    :return:
    """
    best_gain = 0
    best_question = None
    curr_impurity = gini_impurity(breakpoint())

    for col in range(examples[0][0]):
        continue


    return best_question


def split_examples(examples, question):
    """
    Split a bucket based on the passed question.
    :param examples: Examples in a bucket.
    :param question: Question to split examples on.
    :return: (list of examples that pass the question, list of ones that don't)
    """
    true_ex = [e for e in examples if question.answer(e)]
    false_ex = [e for e in examples if e not in true_ex]

    return true_ex, false_ex


def build_tree(examples):
    # Return leaf if the current bucket has only type of label
    if is_homogeneous(examples):
        return Leaf(examples)

    # Find best question to split current bucket. Split examples
    split_question = find_split_question(examples)
    true_ex, false_ex = split_examples(examples, split_question)

    # Recursively build the true and false branches with new buckets
    true_branch = build_tree(true_ex)
    false_branch = build_tree(false_ex)

    return TreeNode(split_question, true_branch, false_branch)


def main():
    # Get mushroom data
    examples = get_mushroom_data.get_data()
    features = [e[0] for e in examples]
    labels = [e[1] for e in examples]

    # Split data into training and test sets
    split = int(len(features)*0.8)
    train_features = features[:split]
    train_labels = labels[:split]
    test_features = features[split:]
    test_labels = labels[split:]


if __name__ == '__main__':
    main()