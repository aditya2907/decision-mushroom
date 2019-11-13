"""
Train a decision tree on a dataset on mushrooms. Test the decision tree in
predicting whether a mushroom is poisonous or edible. The decision tree here is
built from scratch.
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
        self.prediction = plurality(examples)


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
    return counts.most_common()[0][0]


def is_homogeneous(examples):
    """
    Tell if the current bucket has only one type of label.
    :param examples: list of examples in the bucket.
    :return: True if there is only one type of label in the bucket, else False
    """
    return len(set([e[1] for e in examples])) == 1


def gini_impurity(examples):
    """
    Calculate the Gini impurity of a bucket
    :param examples: list of examples in a bucket
    :return: Gini impurity of the bucket (float)
    """
    gini = 0
    label_counts = Counter(e[1] for e in examples)

    for l in label_counts:
        p = label_counts[l]/len(examples)
        gini += p*(1 - p)

    return gini


def info_gain(bucket_1, bucket_2, cur_gini):
    """
    Calculate the information gain of bucket split
    :param bucket_1: list of examples in one branch
    :param bucket_2: list of examples in the other branch
    :param cur_gini: Gini impurity before the proposed split
    :return: difference in Gini impurity before and after the split
    """
    p = len(bucket_1)/(len(bucket_1) + len(bucket_2))
    new_gini = p*gini_impurity(bucket_1) + (1-p)*gini_impurity(bucket_2)

    return cur_gini - new_gini


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


def find_split_question(examples):
    """
    Find the best question to split a bucket with.
    :param examples: list of examples in the bucket.
    :return:
    """
    # Variables to track best question
    best_gain = 0
    best_question = None
    curr_impurity = gini_impurity(examples)

    # Find question feature by feature
    for col in range(len(examples[0][0])):
        # Get possible values for the feature
        feature_vals = set([e[0][col] for e in examples])

        # Test a question based on each feature value
        for v in feature_vals:
            question = Question(lambda x: x == v, col)
            true_ex, false_ex = split_examples(examples, question)

            if (not true_ex) or (not false_ex):
                continue

            cur_gain = info_gain(true_ex, false_ex, curr_impurity)

            # Update best question
            if cur_gain > best_gain:
                best_gain = cur_gain
                best_question = question

    return best_question, best_gain


def build_tree(examples):
    """
    Build a decision tree trained on passed data.
    :param examples: list of pairs of features and labels
    :return: the top node of a decision tree
    """
    # Return leaf if the current bucket has only type of label
    if is_homogeneous(examples):
        return Leaf(examples)

    # Find best question to split current bucket. Split examples
    split_question, gain = find_split_question(examples)
    true_ex, false_ex = split_examples(examples, split_question)

    # Skip if the split leads to an empty branch
    if not true_ex or not false_ex:
        return Leaf(examples)

    # Recursively build the true and false branches with new buckets
    true_branch = build_tree(true_ex)
    false_branch = build_tree(false_ex)

    return TreeNode(split_question, true_branch, false_branch)


def classify(tree_node, example):
    """
    Give a prediction of the label of the given example.
    :param tree_node: top node of the decision tree
    :param example: the example to be classified
    """
    if isinstance(tree_node, Leaf):
        return tree_node.prediction

    if tree_node.question.answer(example):
        classify(tree_node.true_branch, example)
    else:
        classify(tree_node.false_branch, example)


def run_test(tree, test_data):
    """
    Predict the labels of a list of data. Display the accuracy of the
    classifier.
    :param tree: a decision tree
    :param test_data: list of pairs of features and labels
    """
    num_correct = 0

    for d in test_data:
        prediction = classify(tree, test_data[0])

        print(f'Actual: {d[1]}    Prediction: {prediction}')

        if prediction == d[1]:
            num_correct += 1

    accuracy = (num_correct/len(test_data))*100
    print(f'Accuracy: {num_correct}/{len(test_data)} ({accuracy}%)')


def main():
    """
    Get mushroom data, split data into training and test sets, train a
    decision tree on training data, test the tree for accuracy
    """
    # Get mushroom data
    examples = get_mushroom_data.get_data()[:1000]

    # Split data into training and test sets
    split = int(len(examples)*0.8)
    train_data = examples[:split]
    test_data = examples[split:]

    # Fit a tree to training data
    tree = build_tree(train_data)
    print(tree)
    # Run a test on the tree
    run_test(tree, test_data)


if __name__ == '__main__':
    main()
