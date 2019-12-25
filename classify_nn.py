"""
Predict whether a mushroom is poisonous or edible using a neural network.
Heikal Badrulhisham, 2019 <heikal93@gmail.com>
"""
import tensorflow as tf
import pandas


def get_datasets():
    """
    Get mushroom data and prepare them for use in a Tensorflow model
    :return: training dataset and testing dataset
    """
    # Get data
    dataframe = pandas.read_csv('agaricus-lepiota.data')

    # Convert data type for all variables
    for column in dataframe:
        dataframe[column] = pandas.Categorical(dataframe[column])
        dataframe[column] = dataframe[column].cat.codes

    # Get labels
    target = dataframe.pop('p')

    # Get tensors, and split data into training and test sets
    split = int(len(dataframe) * 0.8)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe[:split].values, target[:split].values))

    train_dataset = train_dataset.shuffle(len(dataframe)).batch(1)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe[split:].values, target[split:].values))

    test_dataset = test_dataset.shuffle(len(dataframe)).batch(1)

    return train_dataset, test_dataset


def main():
    """
    Get datasets, build and train a neural network model and test said model
    on unseen data.
    """
    # Get datasets
    train_dataset, test_dataset = get_datasets()

    # Build neural network
    layers = [tf.keras.layers.Dense(22, activation='sigmoid'),
              tf.keras.layers.Dense(30, activation='sigmoid'),
              tf.keras.layers.Dense(1, activation='sigmoid')]

    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=10)

    # Test model
    model.evaluate(test_dataset, verbose=2)


if __name__ == '__main__':
    main()
    exit(0)
