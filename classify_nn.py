import tensorflow as tf
import pandas


def main():
    dataframe = pandas.read_csv('agaricus-lepiota.data')

    for column in dataframe:
        dataframe[column] = pandas.Categorical(dataframe[column])
        dataframe[column] = dataframe[column].cat.codes

    target = dataframe.pop('p')

    split = int(len(dataframe)*0.8)
    train_dataset = tf.data.Dataset.from_tensor_slices((dataframe[:split].values, target[:split].values))
    train_dataset = train_dataset.shuffle(len(dataframe)).batch(1)

    test_dataset = tf.data.Dataset.from_tensor_slices((dataframe[split:].values, target[split:].values))
    test_dataset = test_dataset.shuffle(len(dataframe)).batch(1)

    layers = [tf.keras.layers.Dense(10, activation='sigmoid'),
              tf.keras.layers.Dense(10, activation='sigmoid'),
              tf.keras.layers.Dense(1, activation='sigmoid')]

    model = tf.keras.models.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=10)

    # Test model
    model.evaluate(test_dataset, verbose=2)


if __name__ == '__main__':
    main()
    exit(0)
