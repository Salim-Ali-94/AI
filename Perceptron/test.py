import numpy as np
import __perceptron as perceptron
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"


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
        print("The new flower belongs to the Setosa species\n")
    elif (label == 1):
        print("The new flower belongs to the Versicolor species\n")

    return neuron

def plot(neuron, new_point):

    minimum_input = 0.9*min(min(neuron.features[:, 0]), min(neuron.test_inputs[:, 0]))
    maximum_input = 1.1*max(max(neuron.features[:, 0]), max(neuron.test_inputs[:, 0]))
    minimum_output = 0.9*min(min(neuron.features[:, 1]), min(neuron.test_inputs[:, 1]))
    maximum_output = 1.1*max(max(neuron.features[:, 1]), max(neuron.test_inputs[:, 1]))
    x = np.linspace(minimum_input - 4, maximum_input + 4, 100)
    y = -(neuron.weights[0] / neuron.weights[1])*x - neuron.bias / neuron.weights[1]
    total = neuron.features.shape[0]
    half = total // 2

    neuron.plotter()
    plt.figure()
    axis = plt.axes(facecolor = "#E6E6E6")
    axis.set_axisbelow(True)
    plt.grid(color = "w", linestyle = "solid")

    for spine in axis.spines.values():
        spine.set_visible(False)

    plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
    plt.tick_params(axis = "y", which = "both", left = False, right = False)
    plt.plot(x, y, color = "black", linewidth = 1, label = "Decision boundary")
    plt.plot(new_point[0], new_point[1], "x", color = "indigo", label = "New point")
    plt.plot(neuron.features[0:half, 0], neuron.features[0:half, 1], ".", color = "red", label = "Species setosa")
    plt.plot(neuron.features[half:total, 0], neuron.features[half:total, 1], ".", color = "blue", label = "Species versicolor")
    plt.plot(neuron.test_inputs[:, 0], neuron.test_inputs[:, 1], ".", color = "green", label = "Test data")
    plt.gca().set_xlim(left = minimum_input, right = maximum_input)
    plt.gca().set_ylim(bottom = minimum_output, top = maximum_output)
    plt.xlabel('Sepal length')
    plt.ylabel('Petal length')
    plt.legend()
    plt.savefig('classifier_results.png', bbox_inches = "tight", dpi = 200)
    plt.show()

if __name__ == "__main__":

    episodes, learning_rate = 500, 0.1
    training_data_percent = 75
    new_point = np.array([5.31, 3.76])
    x, y = initialize()
    node = predict(x, y, learning_rate, episodes, training_data_percent, new_point)
    plot(node, new_point)
