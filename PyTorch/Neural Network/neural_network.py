import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as NN
import torch.optim as solver
from torch.utils.data import DataLoader as loader
from torch.utils.data import TensorDataset as group
from sklearn.model_selection import train_test_split as split
import torchvision.transforms as transformation
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"


utility = {"nll": "NLLLoss", "bce": "BCELoss", "mse": "MSELoss", "crossentropy": "CrossEntropyLoss"}
optimization = {"adam": "Adam", "rms": "RMSProp", "sgd": "SGD"}
dataset = {"mnist": "MNIST", "cifar": "CIFAR10", "celeb": "CelebA", "fashion": "FashionMNIST"}
criterion = lambda score: getattr(torch.nn, score)()
algorithm = lambda model, method, learning_rate: getattr(torch.optim, method)(model.parameters(), lr = learning_rate)
library = lambda family, folder: getattr(torchvision.datasets, family)(root = folder, transform = transformation.ToTensor(), download = True)
normalize = lambda data: (data - data.min()) / (data.max() - data.min())

class ArtificialNeuralNetwork(NN.Module):

	activation = {"relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid", "softmax": "Softmax", "logsoftmax": "LogSoftmax"}
	function = lambda self, transform: getattr(torch.nn, transform)(dim = 1) if ("Softmax" in transform) else getattr(torch.nn, transform)()
	forward = lambda self, data: self.network(data)

	def __init__(self, nodes, functions):

		super().__init__()
		depth = len(nodes) - 1
		self.network = NN.Sequential()

		for index in range(depth):

			self.network.add_module(f"layer {index + 1}", NN.Linear(nodes[index], nodes[index + 1]))
			if (type(functions[index]) != str): self.network.add_module(f"activity {index + 1}", self.function(self.activation[functions[index][0].lower().rstrip().lstrip()])) if ((functions[index][0].lower().rstrip().lstrip() in self.activation) & (functions[index][0].lower().rstrip().lstrip() != "")) else self.network.add_module(f"activity {index + 1}", self.function(self.activation["tanh"])), self.network.add_module(f"drop {index + 1}", torch.nn.Dropout(functions[index][1] / 100))
			elif (functions[index].lower().rstrip().lstrip() in self.activation): self.network.add_module(f"activity {index + 1}", self.function(self.activation[functions[index].lower().rstrip().lstrip()]))
			elif (functions[index].lower().rstrip().lstrip() != ""): self.network.add_module(f"activity {index + 1}", self.function(self.activation["tanh"]))




def extract(file, directory = None, encoding = None, output = None, label = None, flag = None):

	if (directory == None):

		ticket = 0 if (label != None) else None
		data = pd.read_csv(f"{file}", header = ticket)
		dimension = len(data.iloc[0]) - 1
		mapping = {encoding[index][0]: encoding[index][1] for index in range(len(encoding))}
		if (label != None): data[f"{label}"] = data[f"{label}"].apply(lambda index: mapping[index])
		else: data.iloc[:, -1] = data.iloc[:, -1].apply(lambda index: mapping[index])
		X = normalize(data.iloc[:, 0:dimension].values)
		if (output > 1): y = data.iloc[:, -1].values
		else: y = data.iloc[:, -1:].values
		if (flag != None): y = normalize(y)

	else: 

		X, Y = [], []
		data = library(dataset[file.lower().rstrip().lstrip()], directory) if (file.lower().rstrip().lstrip() in dataset) else library(dataset["mnist"], directory)
		Z = loader(dataset = data, batch_size = 1)
		for x, y in Z: X.append(x), Y.append(y)
		X, y = torch.cat(X, dim = 0), torch.cat(Y, dim = 0)
		X = X / torch.max(X)
		if (flag != None): y = y / torch.max(y)

	return X, y


def partition(characteristic, category, output, batch, training_percentage, validation_percentage = 0):

	train_percent = training_percentage / 100
	validate_percent = validation_percentage / 100
	test_percent = 1 - (train_percent + validate_percent)
	if ((output == 1) & (isinstance(category[0], np.ndarray) == False)): category = category.reshape((len(category), -1))
	training_x, training_y = characteristic, category
	train_x = torch.FloatTensor(training_x)
	if (output > 1): train_y = torch.LongTensor(training_y)
	else: train_y = torch.FloatTensor(training_y)
	train = group(train_x, train_y)
	trainer = loader(dataset = train, batch_size = batch, shuffle = True)
	tester, validater = None, None

	if (test_percent > 0):

		training_x, testing_x, training_y, testing_y = split(characteristic, category, test_size = test_percent, random_state = np.random.randint(1, 100), shuffle = True, stratify = category)
		train_x = torch.FloatTensor(training_x)
		if (output > 1): train_y = torch.LongTensor(training_y)
		else: train_y = torch.FloatTensor(training_y)
		train = group(train_x, train_y)
		trainer = loader(dataset = train, batch_size = batch, shuffle = True)
		test_x = torch.FloatTensor(testing_x)
		if (output > 1): test_y = torch.LongTensor(testing_y)
		else: test_y = torch.FloatTensor(testing_y)
		test = group(test_x, test_y)
		tester = loader(dataset = test, batch_size = batch, shuffle = True)

	if (validate_percent > 0):

		training_x, validation_x, training_y, validation_y = split(training_x, training_y, test_size = validate_percent, random_state = np.random.randint(1, 100), shuffle = True, stratify = training_y)
		train_x = torch.FloatTensor(training_x)
		if (output > 1): train_y = torch.LongTensor(training_y)
		else: train_y = torch.FloatTensor(training_y)
		train = group(train_x, train_y)
		trainer = loader(dataset = train, batch_size = batch, shuffle = True)
		validate_x = torch.FloatTensor(validation_x)
		if (output > 1): validate_y = torch.LongTensor(validation_y)
		else: validate_y = torch.FloatTensor(validation_y)
		validate = group(validate_x, validate_y)
		validater = loader(dataset = validate, batch_size = batch, shuffle = True)

	return trainer, tester, validater


def learn(trainer, neurons, functions, learning_rate, episodes, propagator, cost):

	collect, score = [], []
	accuracy, ratio = [], []
	correct, incorrect, Y = 0, 0, []
	for x, y in trainer: Y += [y[index].item() for index in range(len(y))]
	labels = list(set(Y))
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if ((neurons[-1] > 1) & (cost.lower().rstrip().lstrip() == "crossentropy") & ("softmax" in functions[-1])): functions[-1] = ""
	ANN = ArtificialNeuralNetwork(neurons, functions).to(device)
	if (propagator.lower().rstrip().lstrip() in optimization): optimizer = algorithm(ANN, optimization[propagator.lower().rstrip().lstrip()], learning_rate)
	else: optimizer = solver.Adam(ANN.parameters(), lr = learning_rate)
	if (cost.lower().rstrip().lstrip() in utility): error = criterion(utility[cost.lower().rstrip().lstrip()])
	elif (neurons[-1] > 1): error = NN.CrossEntropyLoss()
	else: error = NN.MSELoss()
	ANN.train()

	for epoch in range(episodes):

		for x, y in trainer:

			x, y = x.to(device), y.to(device)
			if (len(x.shape) > 2): x = x.reshape(x.shape[0], -1)
			optimizer.zero_grad()
			prediction = ANN(x)
			loss = error(prediction, y)
			loss.backward()
			optimizer.step()
			collect.append(loss.item())

			for index in range(len(prediction)):

				if (neurons[-1] > 1): 

					if (torch.argmax(prediction[index]) == y[index]): correct += 1 
					else: incorrect += 1

				elif (neurons[-1] == 1): 

					if (min(labels, key = lambda x: abs(x - prediction[index].item())) == y[index].item()): correct += 1
					else: incorrect += 1

			total = correct + incorrect
			ratio.append(correct / total)
			correct, incorrect = 0, 0

		score.append(sum(collect) / len(collect))
		accuracy.append(sum(ratio) / len(ratio))
		print("Episode:", epoch + 1)
		print("Error:", round(score[-1], 4))
		print("Accuracy:", round(accuracy[-1], 4), "\n")
		collect, ratio = [], []
		correct, incorrect = 0, 0

	return ANN, score, accuracy


def plot(data, colour, name, x, y):

	episodes = range(1, len(data) + 1)
	plt.figure()
	axis = plt.axes(facecolor = "#E6E6E6")
	axis.set_axisbelow(True)
	plt.grid(color = "w", linestyle = "solid")
	for spine in axis.spines.values(): spine.set_visible(False)
	plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
	plt.tick_params(axis = "y", which = "both", left = False, right = False)
	plt.plot(episodes, data, color = f"{colour}", linewidth = 1)
	plt.xlabel(f"{x}")
	plt.ylabel(f"{y}")
	plt.savefig(f"{name}.png", dpi = 200)
	plt.show()


def test(model, data, output):

	correct, incorrect, Y = 0, 0, []
	for x, y in data: Y += [y[index].item() for index in range(len(y))]
	labels = list(set(Y))
	model.eval()

	with torch.no_grad():

		for x, y in data:

			if (len(x.shape) > 2): x = x.reshape(x.shape[0], -1)
			prediction = model(x)

			for index in range(len(prediction)):

				if (output > 1): 

					if (torch.argmax(prediction[index]) == y[index]): correct += 1 
					else: incorrect += 1

				elif (output == 1): 

					if (min(labels, key = lambda x: abs(x - prediction[index].item())) == y[index].item()): correct += 1
					else: incorrect += 1

	total = correct + incorrect
	if (incorrect == 1): print(f"Correctly labeled {correct} samples and incorrectly labeled {incorrect} sample")
	elif (correct == 1): print(f"Correctly labeled {correct} sample and incorrectly labeled {incorrect} samples")
	else: print(f"Correctly labeled {correct} samples and incorrectly labeled {incorrect} samples")
	print("Accuracy: ", round(100*(correct / total), 2))
