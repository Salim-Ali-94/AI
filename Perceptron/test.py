import numpy as np
import perceptron
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # setup the variables
    episodes, learning_rate = 100, 0.01
    train_percent = 75
    new_point = np.array([5.31, 3.76])
    data = pd.read_csv('iris.csv')
    y = data.iloc[0:100, 4].values
    y = np.where(y == 'setosa', 0, 1)
    X = data.iloc[0:100, [0, 2]].values

	# apply the classifier
    ANN = perceptron.Perceptron(X, y, learning_rate, episodes)
    ANN.splitter(train_percent)
    ANN.train()
    ANN.tester()
    label = int(ANN.classifier(new_point))

    if (label == 0):
        print("\nThe new flower belongs to the Setosa species\n")
    elif (label == 1):
        print("\nThe new flower belongs to the Versicolor species\n")

    # plot the results
    ANN.plotter()
    x = np.linspace(-1, 10, 1000)
    y = -(ANN.weights[0] / ANN.weights[1])*x - ANN.bias / ANN.weights[1]
    total = ANN.features.shape[0]
    half = total // 2
    plot_name = "classifier_results"

    plt.figure()
    plt.plot(x, y, color = "black", label = "$Decision $ $boundary$")
    plt.plot(new_point[0], new_point[1], "x", color = "indigo", label = "$New $ $point$")
    plt.plot(ANN.features[0:half, 0], ANN.features[0:half, 1], ".", color = "red", label = "$Species $ $setosa$")
    plt.plot(ANN.features[half:total, 0], ANN.features[half:total, 1], ".", color = "blue", label = "$Species $ $versicolor$")
    plt.plot(ANN.test_inputs[:, 0], ANN.test_inputs[:, 1], ".", color = "green", label = "$Test $ $data$")
    plt.gca().set_xlim(left = 0.9*X[0:100, 0].min(), right = 1.1*X[0:100, 0].max())
    plt.gca().set_ylim(bottom = 0.9*X[0:100, 1].min(), top = 1.1*X[0:100, 1].max())
    plt.xlabel('$Sepal $ $length$')
    plt.ylabel('$Petal $ $length$')
    plt.legend()
    plt.savefig('{}.png'.format(plot_name), bbox_inches = "tight", dpi = 200)    
    plt.show()
