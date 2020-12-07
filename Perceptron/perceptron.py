import numpy as np
import matplotlib.pyplot as plt


class Perceptron(object):

	sigmoid = lambda self, x: 1 / (1 + np.exp(-x))
	activation = lambda self, features, weights, bias: self.sigmoid(weights.dot(features) + bias)
	classifier = lambda self, inputs: round(self.activation(inputs, self.weights, self.bias))

	def __init__(self, features, labels, learning_rate = 0.1, epochs = 100):

		self.features = features
		self.labels = labels
		self.learning_rate = learning_rate
		self.epochs = epochs
		self.weights = np.random.uniform(-1, 1, self.features.shape[1])
		self.bias = np.random.uniform(-1, 1)
		self.indicator = 0

	def train(self):

		self.totalError = []
		Error = 0

		for episode in range(self.epochs):

			for example in range(self.features.shape[0]):

				output = self.activation(self.features[example], self.weights, self.bias)
				error = self.labels[example] - output
				update = self.learning_rate*error
				self.weights += update*self.features[example]
				self.bias += update
				Error += error

			self.totalError.append(Error)
			Error = 0

		self.totalError = np.asarray(self.totalError)

	def tester(self, data = None, target = None):

		miss_classification = 0
		sucessful_classification = 0

		if (self.indicator == 0):

			for index in range(data.shape[0]):

				output = int(self.classifier(data[index]))

				if (output == target[index]):
					sucessful_classification += 1
				elif (output != target[index]):
					miss_classification += 1

		elif (self.indicator != 0):

			for index in range(self.test_inputs.shape[0]):

				output = int(self.classifier(self.test_inputs[index]))

				if (output == self.test_outputs[index]):
					sucessful_classification += 1
				elif (output != self.test_outputs[index]):
					miss_classification += 1

		print("The classifier correctly labeled {} input samples "\
		 	  "and incorrectly labeled {} samples from the test "\
		 	  "dataset\n".format(sucessful_classification, miss_classification))

	def plotter(self):

		episodes = range(1, self.epochs + 1)
		plt.figure()
		plt.plot(episodes, self.totalError, color = "blue")
		plt.xlabel('$Episode$')
		plt.ylabel('$Total $ $error $ $per $ $episode$')
		plt.savefig('accumulated_errors_for_each_epoch.png', bbox_inches = "tight", dpi = 200)    
		plt.show()

	def splitter(self, percentage):

		factor = percentage / 100
		total = len(self.labels)
		half = total // 2
		fraction = int(half*factor)
		remainder = int(total - (half - fraction))
		trainingSet_labelA = self.labels[0:fraction]
		trainingSet_labelB = self.labels[half:remainder]
		testSet_labelA = self.labels[fraction:half]
		testSet_labelB = self.labels[remainder:total]
		trainingSet_outputs = np.concatenate((trainingSet_labelA, trainingSet_labelB))
		testSet_outputs = np.concatenate((testSet_labelA, testSet_labelB))
		trainingSet_characteristicA = self.features[0:fraction, :]
		trainingSet_characteristicB = self.features[half:remainder, :]
		testSet_characteristicA = self.features[fraction:half, :]
		testSet_characteristicB = self.features[remainder:total, :]
		trainingSet_inputs = np.concatenate((trainingSet_characteristicA, trainingSet_characteristicB))
		testSet_inputs = np.concatenate((testSet_characteristicA, testSet_characteristicB))

		self.features = trainingSet_inputs
		self.labels = trainingSet_outputs
		self.test_inputs = testSet_inputs
		self.test_outputs = testSet_outputs
		self.indicator = 1
