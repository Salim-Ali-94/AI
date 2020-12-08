import numpy as np
import perceptron
import pandas as pd
import matplotlib.pyplot as plt


def initialize():

    data = pd.read_csv('iris.csv')
    classes = data.iloc[0:100, 4].values
    classes = np.where(classes == 'setosa', 0, 1)
    characteristics = data.iloc[0:100, [0, 2]].values

    return characteristics, classes

def predict(inputs, outputs, learning_rate, episodes, training_data_percent, new_point):

    neuron = perceptron.Perceptron(inputs, outputs, learning_rate, episodes)
    neuron.splitter(training_data_percent)
    neuron.train()
    neuron.tester()
    label = int(neuron.classifier(new_point))

    if (label == 0):
        print("\nThe new flower belongs to the Setosa species\n")
    elif (label == 1):
        print("\nThe new flower belongs to the Versicolor species\n")

    neuron.plotter()

    return neuron.weights, neuron.bias, neuron.features, neuron.test_inputs

def plot(weights, bias, training_data, test_data, new_point):

    minimum_input = 0.9*min(min(training_data[:, 0]), min(test_data[:, 0]))
    maximum_input = 1.1*max(max(training_data[:, 0]), max(test_data[:, 0]))
    minimum_output = 0.9*min(min(training_data[:, 1]), min(test_data[:, 1]))
    maximum_output = 1.1*max(max(training_data[:, 1]), max(test_data[:, 1]))
    x = np.linspace(minimum_input - 4, maximum_input + 4, 100)
    y = -(weights[0] / weights[1])*x - bias / weights[1]
    total = training_data.shape[0]
    half = total // 2

    plt.figure()
    plt.plot(x, y, color = "black", label = "$Decision $ $boundary$")
    plt.plot(new_point[0], new_point[1], "x", color = "indigo", label = "$New $ $point$")
    plt.plot(training_data[0:half, 0], training_data[0:half, 1], ".", color = "red", label = "$Species $ $setosa$")
    plt.plot(training_data[half:total, 0], training_data[half:total, 1], ".", color = "blue", label = "$Species $ $versicolor$")
    plt.plot(test_data[:, 0], test_data[:, 1], ".", color = "green", label = "$Test $ $data$")
    plt.gca().set_xlim(left = minimum_input, right = maximum_input)
    plt.gca().set_ylim(bottom = minimum_output, top = maximum_output)
    plt.xlabel('$Sepal $ $length$')
    plt.ylabel('$Petal $ $length$')
    plt.legend()
    plt.savefig('classifier_results.png', bbox_inches = "tight", dpi = 200)    
    plt.show()

if __name__ == "__main__":

    episodes, learning_rate = 100, 0.01
    training_data_percent = 75
    new_point = np.array([5.31, 3.76])
    x, y = initialize()
    weights, bias, training_data, test_data = predict(x, y, learning_rate, episodes, training_data_percent, new_point)
    plot(weights, bias, training_data, test_data, new_point)
