# Can we eat this mushroom?

This project is based on a dataset on mushrooms consisting of physical
attributes such as cap shape and odor and each mushroom in the
sample is classified as either edible or poisonous. The goal here is to 
train decision trees to predict whether a mushroom is edible or poisonous
based on its physical attributes. 

`decision_tree_scikit.py` uses a decision
tree from `sklearn`. `decision_tree_scikit.py` uses a decision tree from
made from scratch. `get_mushroom_data.py` is used in both of the aforementioned
programs to load the mushroom, including extrapolating missing feature values.
The decision tree made from scratch achieved 100% accuracy in multiple tests.

## Data
Data source: http://archive.ics.uci.edu/ml/datasets/Mushroom

Dataset in this repository: agaricus-lepiota.data

Description of the dataset: agaricus-lepiota.names

## Requirements
- scikit-learn