"""
Train a decision tree on a dataset on mushrooms using sklearn. Test the
decision tree in predicting whether a mushroom is poisonous or edible.
Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
import get_mushroom_data
from sklearn.tree import DecisionTreeClassifier
import random


def main():
    # Get mushroom data
    examples = get_mushroom_data.get_data(convert=True)
    random.shuffle(examples)
    features = [e[0] for e in examples]
    labels = [e[1] for e in examples]

    # Split data into training and test sets
    split = int(len(features)*0.8)
    train_features = features[:split]
    train_labels = labels[:split]
    test_features = features[split:]
    test_labels = labels[split:]

    # Train decision tree on training data
    decision_tree = DecisionTreeClassifier()
    decision_tree = decision_tree.fit(train_features, train_labels)

    # Test decision tree on test data
    # Iterative testing
    for i in range(len(test_features)):
        print(f'Features: {test_features[i]}    '
              f'Actual label: {test_labels[i]}  '
              f'Predicted label: {decision_tree.predict([test_features[i]])}')

    # Show accuracy over test set
    accuracy = decision_tree.score(test_features, test_labels)
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    main()
    exit(0)
