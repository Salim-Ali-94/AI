import numpy as np
import perceptron
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # setup the variables
    episodes, learning_rate = 100, 0.01
    training_data_percent = 75
    new_point = np.array([5.31, 3.76])
    data = pd.read_csv('iris.csv')
    y = data.iloc[0:100, 4].values
    y = np.where(y == 'setosa', 0, 1)
    X = data.iloc[0:100, [0, 2]].values

    # apply the classifier
    neuron = perceptron.Perceptron(X, y, learning_rate, episodes)
    neuron.splitter(training_data_percent)
    neuron.train()
    neuron.tester()
    label = int(neuron.classifier(new_point))

    if (label == 0):
        print("\nThe new flower belongs to the Setosa species\n")
    elif (label == 1):
        print("\nThe new flower belongs to the Versicolor species\n")

    # plot the results
    neuron.plotter()
    x = np.linspace(0.9*X[0:100, 0].min() - 4, 1.1*X[0:100, 0].max() + 4, 100)
    y = -(neuron.weights[0] / neuron.weights[1])*x - neuron.bias / neuron.weights[1]
    total = neuron.features.shape[0]
    half = total // 2

    plt.figure()
    plt.plot(x, y, color = "black", label = "$Decision $ $boundary$")
    plt.plot(new_point[0], new_point[1], "x", color = "indigo", label = "$New $ $point$")
    plt.plot(neuron.features[0:half, 0], neuron.features[0:half, 1], ".", color = "red", label = "$Species $ $setosa$")
    plt.plot(neuron.features[half:total, 0], neuron.features[half:total, 1], ".", color = "blue", label = "$Species $ $versicolor$")
    plt.plot(neuron.test_inputs[:, 0], neuron.test_inputs[:, 1], ".", color = "green", label = "$Test $ $data$")
    plt.gca().set_xlim(left = 0.9*X[0:100, 0].min(), right = 1.1*X[0:100, 0].max())
    plt.gca().set_ylim(bottom = 0.9*X[0:100, 1].min(), top = 1.1*X[0:100, 1].max())
    plt.xlabel('$Sepal $ $length$')
    plt.ylabel('$Petal $ $length$')
    plt.legend()
    plt.savefig('classifier_results.png', bbox_inches = "tight", dpi = 200)    
    plt.show()
