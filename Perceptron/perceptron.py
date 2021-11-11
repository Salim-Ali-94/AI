import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"


class Perceptron(object):

	sigmoid = lambda self, x: 1 / (1 + np.exp(-x))
	tanh = lambda self, x: np.tanh(x)
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
		self.cost = []


	def partition(self, percentage):

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
		self.features = np.copy(trainingSet_inputs)
		self.labels = np.copy(trainingSet_outputs)
		self.test_inputs = np.copy(testSet_inputs)
		self.test_outputs = np.copy(testSet_outputs)
		self.indicator = 1


	def learn(self):

		loss = 0

		for episode in range(self.epochs):

			for example in range(self.features.shape[0]):

				output = self.activation(self.features[example], self.weights, self.bias)
				error = self.labels[example] - output
				update = self.learning_rate*error
				self.weights += update*self.features[example]
				self.bias += update
				loss += error

			print("Episode:", episode + 1)
			print("Error:", loss, "\n\n")
			self.cost.append(loss)
			loss = 0


	def test(self, data = None, target = None):

		incorrect = 0
		correct = 0

		if (self.indicator == 0):

			for index in range(data.shape[0]):

				output = int(self.classifier(data[index]))

				if (output == target[index]): correct += 1
				elif (output != target[index]): incorrect += 1

		elif (self.indicator != 0):

			for index in range(self.test_inputs.shape[0]):

				output = int(self.classifier(self.test_inputs[index]))
				if (output == self.test_outputs[index]): correct += 1
				elif (output != self.test_outputs[index]): incorrect += 1

		if (incorrect == 1): print(f"The classifier correctly labeled {correct} input samples and incorrectly labeled {incorrect} sample from the test dataset\n\n")
		elif (correct == 1): print(f"The classifier correctly labeled {correct} input sample and incorrectly labeled {incorrect} samples from the test dataset\n\n")
		else: print(f"The classifier correctly labeled {correct} input samples and incorrectly labeled {incorrect} samples from the test dataset\n\n")


	def plot(self):

		episodes = range(1, self.epochs + 1)
		plt.figure()
		axis = plt.axes(facecolor = "#E6E6E6")
		axis.set_axisbelow(True)
		plt.grid(color = "w", linestyle = "solid")
		for spine in axis.spines.values(): spine.set_visible(False)
		plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
		plt.tick_params(axis = "y", which = "both", left = False, right = False)
		plt.plot(episodes, self.cost, color = "blue", linewidth = 1)
		plt.xlabel('Episode')
		plt.ylabel('Total error per episode')
		plt.savefig('accumulated_errors_over_each_epoch.png', dpi = 200)
		plt.show()
