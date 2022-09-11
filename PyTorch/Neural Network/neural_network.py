from PIL import Image
import pandas as pd
import numpy as np
from math import factorial
import itertools, cv2, imutils
import torch, torchvision, os, re, sys
import torch.nn as NN
import torch.optim as solver
from torch.utils.data import DataLoader as loader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset as group
from sklearn.model_selection import train_test_split as split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transformation
import torch.nn.functional as graph
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from torchsummary import summary
import torch.nn.init
import collections, datetime, time
from sklearn import preprocessing
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import subprocess
import optuna
import pygame as pg
plt.rcParams["font.family"] = "Arial"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


activation = { "relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid", "softmax": "Softmax", "logsoftmax": "LogSoftmax", "leaky": "LeakyReLU" }
utility = { "nll": "NLLLoss", "bce": "BCELoss", "mse": "MSELoss", "ce": "CrossEntropyLoss", "bcel": "BCEWithLogitsLoss" }
optimization = { "adam": "Adam", "rms": "RMSProp", "sgd": "SGD", "lbfgs": "LBFGS" }
dataset = { "mnist": "MNIST", "cifar": "CIFAR10", "celeb": "CelebA", "fashion": "FashionMNIST", "emnist": "EMNIST" }
normalizer = { "batch": "BatchNorm2d", "instance": "InstanceNorm2d", "layer": "LayerNorm2d" }
pool = { "avg": "AvgPool2d", "max": "MaxPool2d" }
initialization = { "ku": "kaiming_uniform_", "kn": "kaiming_normal_", "xu" : "xavier_uniform_", "xn" : "xavier_normal_" }
expand = { "nll": "Negative Log Liklihood", "bce": "Binary Cross-Entropy", "mse": "Mean Square Error", "ce": "Cross-Entropy", "bcel": "Binary Cross-Entropy (with Logits)", "avg": "Average Pooling", "max": "Max Pooling", "batch": "Batch Norm", "layer": "Layer Norm", "instance": "Instance Norm", "sgd": "Stochastic Gradient Decent", "adam": "Adam", "rms": "Root Mean Square Propagation", "lbfgs": "Limited-memory Broyden Fletcher Goldfarb Shanno" }
value = { "tanh": -1, "sigmoid": 0, "relu": 0, "": 0 }
function = lambda transform, slope = None: getattr(torch.nn, transform)(dim = 1) if ("Softmax" in transform) else getattr(torch.nn, transform)(slope) if (("LeakyReLU" in transform) & (slope != None)) else getattr(torch.nn, transform)()
criterion = lambda score: getattr(torch.nn, score[0])(ignore_index = score[1]) if (type(score) != str) else getattr(torch.nn, score)()
algorithm = lambda model, method, learning_rate, momentum = 0, beta = (): getattr(torch.optim, method)(model.parameters(), lr = learning_rate, betas = beta) if (beta != ()) else getattr(torch.optim, method)(model.parameters(), lr = learning_rate, momentum = momentum) if (momentum != 0) else getattr(torch.optim, method)(model.parameters(), lr = learning_rate)
library = lambda family, folder, convert: getattr(torchvision.datasets, family)(root = folder, transform = convert, download = True)
normalize = lambda data: (data - data.min()) / (data.max() - data.min())
standardize = lambda data, mean, standard_deviation: (data - mean) / standard_deviation
aggregate = lambda transform, dimension: getattr(torch.nn, transform)(dimension, affine = True) if ("Instance" in transform) else getattr(torch.nn, transform)(dimension)
sampling = lambda transform, window: getattr(torch.nn, transform)(window)
initialize = lambda method, weight, transform = "relu": getattr(torch.nn.init, method)(weight, nonlinearity = transform) if ("kaiming" in method) else getattr(torch.nn.init, method)(weight)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ArtificialNeuralNetwork(NN.Module):

	def __init__(self, nodes, functions, system = None, flatten = False, unflatten = False, channels = 1, width = 28, height = 28, normalization = None):

		super().__init__()
		depth = len(nodes) - 1
		normalization = refresh(normalization)
		if (normalization == []): normalization = [""]*depth
		elif (len(normalization) < depth): normalization = [normalization[index] if (index < len(normalization)) else "" for index in range(depth)]
		self.network = system if (system != None) else NN.Sequential()

		for index in range(depth):

			if ((index == 0) & (flatten == True)):

				self.network.add_module("transform", NN.Flatten())

			if ((type(nodes[index]) == int) & (type(nodes[index + 1]) == int)):

				self.network.add_module(f"layer {index + 1}", NN.Linear(nodes[index], nodes[index + 1]))

			elif ((type(nodes[index]) != int) & (type(nodes[index + 1]) == int)): 

				self.network.add_module(f"layer {index + 1}", NN.Linear(nodes[index][0], nodes[index + 1]))

			elif ((type(nodes[index]) == int) & (type(nodes[index + 1]) != int)): 

				self.network.add_module(f"layer {index + 1}", NN.Linear(nodes[index], nodes[index + 1][0]))

			else: 

				self.network.add_module(f"layer {index + 1}", NN.Linear(nodes[index][0], nodes[index + 1][0]))

			if ((type(nodes[index]) != int) | ((len([item for item in nodes if type(item) != int]) > 0) & (index == 0) & (len(nodes[0][0] if (type(neurons[0]) != int) else nodes[0]) >= 100))):

				if ((type(nodes[index]) == int) & (len(nodes[0][0] if (type(neurons[0]) != int) else nodes[0]) >= 100)):

					nodes[index] = (nodes[index], [item for item in nodes if type(item) != int][-1][1])

				if ((nodes[index][1].lower().lstrip().rstrip() in initialization) & ("k" not in nodes[index][1].lower().lstrip().rstrip())):
	
					initialize(initialization[nodes[index][1].lower().lstrip().rstrip()], list(self.named_parameters())[0][-1])

				elif ((nodes[index][1].lower().lstrip().rstrip() in initialization) & ("k" in nodes[index][1].lower().lstrip().rstrip()) & (type(functions[index]) == str) & (functions[index].lower().lstrip().rstrip() != "softmax") & (functions[index].lower().lstrip().rstrip() != "")):

						if (functions[index].lower().lstrip().rstrip() in activation): initialize(initialization[nodes[index][1].lower().lstrip().rstrip()], list(self.named_parameters())[0][-1], activation[functions[index].lower().lstrip().rstrip()].lower())
						elif (index != depth - 1): initialize(initialization[nodes[index][1].lower().lstrip().rstrip()], list(self.named_parameters())[0][-1], activation["tanh"])
						else: initialize(initialization[nodes[index][1].lower().lstrip().rstrip()], list(self.named_parameters())[0][-1], activation["sigmoid"])

				elif ((nodes[index][1].lower().lstrip().rstrip() in initialization) & ("k" in nodes[index][1].lower().lstrip().rstrip()) & (type(functions[index]) != str) & (functions[index][0].lower().lstrip().rstrip() != "softmax") & (functions[index][0].lower().lstrip().rstrip() != "")):

					if (functions[index][0].lower().lstrip().rstrip() in activation): initialize(initialization[nodes[index][1].lower().lstrip().rstrip()], list(self.named_parameters())[0][-1], activation[functions[index][0].lower().lstrip().rstrip()].lower())
					elif (index != depth - 1): initialize(initialization[nodes[index][1].lower().lstrip().rstrip()], list(self.named_parameters())[0][-1], activation["tanh"])
					else: initialize(initialization[nodes[index][1].lower().lstrip().rstrip()], list(self.named_parameters())[0][-1], activation["sigmoid"])

				else:

					initialize(initialization["xn"], list(self.named_parameters())[0][-1])

			if ((normalization[index].lower().rstrip().lstrip() != "") & (position == 0)):

				if (normalization[index].lower().lstrip().rstrip() in normalizer): self.network.add_module(f"norm {index + 1}", aggregate(normalizer[normalization[index].lower().lstrip().rstrip()], nodes[index + 1]))
				else: self.network.add_module(f"norm {index + 1}", aggregate(normalizer["batch"], nodes[index + 1]))

			if (type(functions[index]) != str): 

				if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 2) & (functions[index][-1] >= 1)): self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()]))
				elif ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) >= 2)): self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()], functions[index][1]))
				elif (functions[index][0].lower().rstrip().lstrip() != ""): self.network.add_module(f"activity {index + 1}", function(activation["tanh"]))
				if (functions[index][-1] >= 1): self.network.add_module(f"drop {index + 1}", torch.nn.Dropout(functions[index][-1] / 100))

			elif (functions[index].lower().rstrip().lstrip() in activation): 

				self.network.add_module(f"activity {index + 1}", function(activation[functions[index].lower().rstrip().lstrip()]))

			elif (functions[index].lower().rstrip().lstrip() != ""): 

				self.network.add_module(f"activity {index + 1}", function(activation["tanh"]))

			if ((normalization[index].lower().rstrip().lstrip() != "") & (position == 1)):

				if (normalization[index].lower().lstrip().rstrip() in normalizer): self.network.add_module(f"norm {index + 1}", aggregate(normalizer[normalization[index].lower().lstrip().rstrip()], nodes[index + 1]))
				else: self.network.add_module(f"norm {index + 1}", aggregate(normalizer["batch"], nodes[index + 1]))

			if ((index == depth - 1) & (unflatten == True)): 

				self.network.add_module("transform", NN.Unflatten(1, (channels, width, height)))


	def forward(self, data): 

		output = self.network(data)
		return output




class ConvolutionalNeuralNetwork(NN.Module):

	def __init__(self, kernel, stride, padding, height, width, convolutions, functions, nodes = None, channels = 1, pooling = None, direction = 1, offset = True, normalization = None, position = 1, flatten = False, unflatten = False):

		super().__init__()
		depth = len(convolutions) - 1
		self.network = NN.Sequential()
		nodes, pooling, normalization = refresh(nodes, pooling, normalization)
		if (normalization == []): normalization = [""]*depth
		elif (len(normalization) < depth): normalization = [normalization[index] if (index < len(normalization)) else "" for index in range(depth)]
		if (pooling == []): pooling = [("", 1)]*depth
		elif (len(pooling) < depth): pooling = [pooling[index] if (index < len(pooling)) else "" for index in range(depth)]
		
		for index in range(depth):

			if (direction == -1): 

				self.network.add_module(f"transpose convolution {index + 1}", NN.ConvTranspose2d(convolutions[index], convolutions[index + 1], kernel[index], stride[index], padding[index], bias = offset))

			else: 

				self.network.add_module(f"convolution {index + 1}", NN.Conv2d(convolutions[index], convolutions[index + 1], kernel[index], stride[index], padding[index], bias = offset))

			if (pooling[index][0].lower().rstrip().lstrip() != ""): 

				if (pooling[index][0].lower().lstrip().rstrip() in pool): self.network.add_module(f"pool {index + 1}", sampling(pool[pooling[index][0].lower().lstrip().rstrip()], pooling[index][1])) 
				else: self.network.add_module(f"pool {index + 1}", sampling(pool["max"], pooling[index][1]))

			if ((normalization[index].lower().rstrip().lstrip() != "") & (position == 0)):

				if (normalization[index].lower().lstrip().rstrip() in normalizer): self.network.add_module(f"norm {index + 1}", aggregate(normalizer[normalization[index].lower().lstrip().rstrip()], convolution[index + 1]))
				else: self.network.add_module(f"norm {index + 1}", aggregate(normalizer["batch"], convolution[index + 1]))

			if (type(functions[index]) != str):

				if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 2) & (functions[index][-1] >= 1)): self.network.add_module(f"activation {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()]))
				elif ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) >= 2)): self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()], functions[index][1]))
				elif (functions[index][0].lower().rstrip().lstrip() != ""): self.network.add_module(f"activity {index + 1}", function(activation["relu"]))
				if (functions[index][-1] >= 1): self.network.add_module(f"drop {index + 1}", torch.nn.Dropout(functions[index][-1] / 100))

			elif (functions[index].lower().rstrip().lstrip() in activation): 

				self.network.add_module(f"activation {index + 1}", function(activation[functions[index].lower().rstrip().lstrip()]))

			elif (functions[index].lower().rstrip().lstrip() != ""): 

				self.network.add_module(f"activation {index + 1}", function(activation["relu"]))

			if ((normalization[index].lower().rstrip().lstrip() != "") & (position == 1)):

				if (normalization[index].lower().lstrip().rstrip() in normalizer): self.network.add_module(f"norm {index + 1}", aggregate(normalizer[normalization[index].lower().lstrip().rstrip()], convolutions[index + 1]))
				else: self.network.add_module(f"norm {index + 1}", aggregate(normalizer["batch"], convolutions[index + 1]))

			if ((direction == 1) & (nodes != [])):

				width = self.dimension(width, kernel[index] if (type(kernel[index]) == int) else kernel[index][0], stride[index] if (type(stride[index]) == int) else stride[index][0], padding[index] if (type(padding[index]) == int) else padding[index][0], pooling[index] if (type(pooling[index]) == int) else pooling[index][1][0])
				height = self.dimension(height, kernel[index] if (type(kernel[index]) == int) else kernel[index][1], stride[index] if (type(stride[index]) == int) else stride[index][1], padding[index] if (type(padding[index]) == int) else padding[index][1], pooling[index] if (type(pooling[index]) == int) else pooling[index][1][1])

			elif ((direction == -1) & (nodes != [])):

				width = self.expand(width, kernel[index] if (type(kernel[index]) == int) else kernel[index][0], stride[index] if (type(stride[index]) == int) else stride[index][0], padding[index] if (type(padding[index]) == int) else padding[index][0], pooling[index] if (type(pooling[index]) == int) else pooling[index][1][0])
				height = self.expand(height, kernel[index] if (type(kernel[index]) == int) else kernel[index][1], stride[index] if (type(stride[index]) == int) else stride[index][1], padding[index] if (type(padding[index]) == int) else padding[index][1], pooling[index] if (type(pooling[index]) == int) else pooling[index][1][1])

			if ((nodes == []) & (flatten == True) & (index == depth - 1)): 

				self.network.add_module("transform", NN.Flatten())

			if ((index == depth - 1) & (nodes != [])):

				if (direction == 1): width = self.dimension(width, kernel[-1] if (type(kernel[-1]) == int) else kernel[-1][0], stride[-1] if (type(stride[-1]) == int) else stride[-1][0], padding[-1] if (type(padding[-1]) == int) else padding[-1][0], pooling[-1][1] if (type(pooling[-1][1]) == int) else pooling[-1][1][0])
				elif (direction == -1): width = self.expand(width, kernel[-1] if (type(kernel[-1]) == int) else kernel[-1][0], stride[-1] if (type(stride[-1]) == int) else stride[-1][0], padding[-1] if (type(padding[-1]) == int) else padding[-1][0], pooling[-1][1] if (type(pooling[-1][1]) == int) else pooling[-1][1][0])
				if (direction == 1): height = self.dimension(height, kernel[-1] if (type(kernel[-1]) == int) else kernel[-1][1], stride[-1] if (type(stride[-1]) == int) else stride[-1][1], padding[-1] if (type(padding[-1]) == int) else padding[-1][1], pooling[-1][1] if (type(pooling[-1][1]) == int) else pooling[-1][1][1])
				elif (direction == -1): height = self.expand(height, kernel[-1] if (type(kernel[-1]) == int) else kernel[-1][1], stride[-1] if (type(stride[-1]) == int) else stride[-1][1], padding[-1] if (type(padding[-1]) == int) else padding[-1][1], pooling[-1][1] if (type(pooling[-1][1]) == int) else pooling[-1][1][1])
				nodes.insert(0, convolutions[-1]*int(width*height))
				ArtificialNeuralNetwork(nodes, functions[depth:], self.network, flatten, unflatten, channels, width, height)


	def forward(self, data): 

		output = self.network(data)
		return output


	def dimension(self, pixels, kernel, stride = 1, padding = 0, pool = 1, dilation = 1): 

		order = np.floor((((pixels + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1) / pool)
		return order


	def expand(self, pixels, kernel, stride = 1, padding = 0, pool = 1, dilation = 1, buffer = 0): 

		order = np.ceil(((pixels - 1)*stride - 2*padding + dilation*(kernel - 1) + buffer + 1) / pool)
		return order




class ImageData(Dataset):

	def __init__(self, csv_folder, image_folder, transform = None, colour = "c", flag = False, row = None):

		data = pd.read_csv(csv_folder, header = row)
		data = data.dropna()
		self.image_folder = image_folder
		self.labels = data.iloc[:, 0].values
		if (flag == False): self.category = data.iloc[:, 1].values
		else: self.category = data.iloc[:, -1:].values
		self.transform = transform
		self.colour = colour.lower().lstrip().rstrip()


	def __getitem__(self, index):

		if (self.colour == "g"): image = Image.open(os.path.join(self.image_folder, self.labels[index])).convert("L")
		elif (self.colour == "a"): image = Image.open(os.path.join(self.image_folder, self.labels[index])).convert("RGBA")
		elif (self.colour == "c"): image = Image.open(os.path.join(self.image_folder, self.labels[index])).convert("RGB")
		else: image = Image.open(os.path.join(self.image_folder, self.labels[index]))
		if (self.transform != None): image = self.transform(image)
		label = self.category[index]
		return image, label


	def __len__(self):

		size = self.category.shape[0]
		return size




def scale(data, minimum, maximum): 

	high = max(data) if (type(data) != torch.Tensor) else data.max().item()
	low = min(data) if (type(data) != torch.Tensor) else data.min().item()
	gradient = fsolve(lambda slope, data, minimum, maximum: ((maximum - minimum + slope*low) / high) - slope, 1, args = (data, minimum, maximum))[0]
	offset = minimum - gradient*low
	data = gradient*data + offset
	return data


def refresh(*args):

	array = []

	for value in args:

		if (value == None): variable = []
		else: variable = value
		array.append(variable)

	return array


def extract(file, directory = None, encoding = None, convert = None, rows = None, columns = None, output = None, label = None, channels = 1, batch = 0, flag = None):

	if ((directory == None) & (batch == 0)):

		encoding, rows, columns = refresh(encoding, rows, columns)
		encoding, replacer = encoding
		ticket = 0 if (label != None) else None
		data = pd.read_csv(file, header = ticket)
		data = truncate(data, list(set(rows)), columns, label)
		dimension = len(data.iloc[0]) - 1
		if (encoding != []): mapping = { encoding[index][0]: encoding[index][1] for index in range(len(encoding)) }, 
						 data.iloc[:, -1] = data.iloc[:, -1].apply(lambda index: mapping[index])
		if (replacer != []): data = replace(data, replacer)
		matrix = data.iloc[:, 0:dimension].values.astype(np.float64)
		if (flag != None): inputs = normalize(matrix)
		else: inputs = standardize(matrix, matrix.mean(), matrix.std())
		if (output > 1): outputs = data.iloc[:, -1].values
		else: outputs = data.iloc[:, -1:].values
		if (flag != None): outputs = normalize(outputs)

	elif (batch == 0):

		inputs, outputs = [], []
		convert = refresh(convert)
		convert.insert(0, transformation.ToTensor())
		if (file.lower().rstrip().lstrip() in dataset): data = library(dataset[file.lower().rstrip().lstrip()], directory, transformation.Compose(list(set(convert))))
		else: data = library(dataset["mnist"], directory, transformation.Compose(convert))
		holder = loader(dataset = data, batch_size = 1, num_workers = 2)
		for x, y in holder: inputs.append(x), outputs.append(y)
		inputs, outputs = torch.cat(inputs, dim = 0), torch.cat(outputs, dim = 0)
		if ((torch.max(inputs) > 255) | (torch.min(inputs) < 0) | (flag != None)): inputs = normalize(inputs)
		else: inputs = standardize(inputs, inputs.mean(), inputs.std())
		if (flag != None): outputs = normalize(outputs)

	return inputs, outputs


def partition(characteristics, categories, output, batch, training_percentage = 100, validation_percentage = 0, cnn = False):

	percent_train = training_percentage / 100
	percent_validate = validation_percentage / 100
	percent_test = round(1 - (percent_train + percent_validate), 1)
	if ((output == 1) & (isinstance(categories[0], np.ndarray) == False) & (cnn == False)): categories = categories.reshape((len(categories), -1))
	training_inputs, training_outputs = characteristics, categories
	train_inputs, train_outputs = torch.FloatTensor(training_inputs), training_outputs
	if (output > 1): train_outputs = torch.LongTensor(training_outputs)
	elif (output == 1): train_outputs = torch.FloatTensor(training_outputs)
	train = group(train_inputs, train_outputs)
	trainer = loader(dataset = train, batch_size = batch, shuffle = True, num_workers = 2)
	tester, validator = [], []

	if (percent_test > 0):

		training_inputs, testing_inputs, training_outputs, testing_outputs = split(characteristics, categories, test_size = percent_test, random_state = np.random.randint(1, 100), shuffle = True, stratify = categories if (1 not in collections.Counter(categories.reshape((len(categories), ))).values()) else None)
		train_inputs = torch.FloatTensor(training_inputs)
		if (output > 1): train_outputs = torch.LongTensor(training_outputs)
		elif (output == 1): train_outputs = torch.FloatTensor(training_outputs)
		else: train_outputs = training_outputs.float()
		train = group(train_inputs, train_outputs)
		trainer = loader(dataset = train, batch_size = batch, shuffle = True, num_workers = 2)
		test_inputs = torch.FloatTensor(testing_inputs)
		if (output > 1): test_outputs = torch.LongTensor(testing_outputs)
		elif (output == 1): test_outputs = torch.FloatTensor(testing_outputs)
		else: test_outputs = testing_outputs.float()
		test = group(test_inputs, test_outputs)
		tester = loader(dataset = test, batch_size = batch, shuffle = True, num_workers = 2)
		duplicate = percent_validate
		percent_validate = (validation_percentage*len(characteristics) / len(train_inputs)) / 100

		if (percent_validate == 0):

			print(), print("*"*120)
			print(), print("DATA PARTITION TABLE")
			print(), print("*"*120)
			print(), print(f"Training data size: {len(train_inputs)} / {len(test_inputs) + len(train_inputs)} ({100*percent_train}%)")
			print(f"Testing data size: {len(test_inputs)} / {len(test_inputs) + len(train_inputs)} ({100*percent_test}%)")
			print(), print("*"*120)

	if (percent_validate > 0):

		training_inputs, validation_inputs, training_outputs, validation_outputs = split(training_inputs, training_outputs, test_size = percent_validate, random_state = np.random.randint(1, 100), shuffle = True, stratify = training_outputs if (1 not in collections.Counter(training_outputs.reshape((len(training_outputs), ))).values()) else None)
		train_inputs = torch.FloatTensor(training_inputs)
		if (output > 1): train_outputs = torch.LongTensor(training_outputs)
		elif (output == 1): train_outputs = torch.FloatTensor(training_outputs)
		else: train_outputs = training_outputs.float()
		train = group(train_inputs, train_outputs)
		trainer = loader(dataset = train, batch_size = batch, shuffle = True, num_workers = 2)
		validate_inputs = torch.FloatTensor(validation_inputs)
		if (output > 1): validate_outputs = torch.LongTensor(validation_outputs)
		elif (output == 1): validate_outputs = torch.FloatTensor(validation_outputs)
		else: validate_outputs = validation_outputs.float()
		validate = group(validate_inputs, validate_outputs)
		validator = loader(dataset = validate, batch_size = batch, shuffle = True, num_workers = 2)
		print(), print("*"*120)
		print(), print("DATA PARTITION TABLE")
		print(), print("*"*120)
		print(), print(f"Training data size: {len(train_inputs)} / {len(test_inputs) + len(train_inputs) + len(validate_inputs)} ({100*percent_train}%)")
		print(f"Validation data size: {len(validate_inputs)} / {len(test_inputs) + len(train_inputs) + len(validate_inputs)} ({round(100*duplicate, 2)}%)")
		print(f"Testing data size: {len(test_inputs)} / {len(test_inputs) + len(train_inputs) + len(validate_inputs)} ({100*test_percent}%)")
		print(), print("*"*120)

	return trainer, tester, validator


def information(ANN, CNN, AE, GAN, DCGAN, DCWGANGP, model, learning_rate, cost, propagator, neurons, convolutions, kernel, stride, padding, pooling, normalization, width, height, channels, dimension, labels, iterations, lamda, validator, horizon, regression):

	print("*"*120), print()
	print(f"MODEL ARCHITECTURE ({'Artificial Neural Network' if ((ANN != []) & (len(neurons) > 2)) else 
				     'Convolutional Neural Network' if (CNN != []) else
				     'Perceptron' if ((ANN != []) & (len(neurons) <= 2)) else
				     'Autoencoder' if (AE != []) else
				     'Generative Adversarial Network' if (GAN != []) else
				     'Deep Convolutional Generative Adversarial Network' if (DCGAN != []) else
				     'Deep Convolutional Wasserstein Generative Adversarial Network + Gradient Penalty' if (DCWGANGP != [])})")
	print(), print("*"*120)

	try: 

		print()
		if (CNN != []): print(summary(model, (channels, height, width)))
		elif ((DCGAN != []) | (DCWGANGP != [])): print(summary(model[0], (channels, height, width)), "\n\n"),
											   print(summary(model[1], (channels, height, width)))
		elif (GAN != []): print(summary(model[0], (1, dimension[0])), "\n\n"),
						  print(summary(model[1], (1, dimension[1])))
		else: print(summary(model, (1, dimension)))

	except Exception as error: 

		print(), print(error), print()
		if ((DCGAN != []) | (DCWGANGP != []) | (GAN != [])): print(model[0], "\n\n"),
														   print(model[1])
		else: print(model)

	print(), print("*"*120)
	print("*"*120), print()
	print(f"SUMMARY OF HYPERPARAMETERS ({'Regression' if (regression == True) else 
					     'Binary Classification' if ((ANN != []) & (size == 1) & (len(labels) == 2)) else
					     'Multi-Class Classification' if ((ANN != []) & (size == 1) & (len(labels) > 2)) else
					     'Multi-Label Classification' if ((ANN != []) & (size > 1) & (len(labels) >= 2)) else
					     'Image Processing' if (CNN != []) else
					     'Data Processing' if (AE != []) else
					     'Data Creation' if ((GAN != []) | (DCGAN != []) | (DCWGANGP != []))})")
	print(), print("*"*120)

	if ((ANN != []) | (AE != [])):

		print()
		if (len(neurons) - 2 > 0): print(f"Number of hidden layers: {len(neurons) - 2}")
		if (len(neurons) - 2 > 0): print("Number of nodes per layer:", " --> ".join([f"[{str(neurons[item + 1]) if (type(neurons[item + 1]) == int) else str(neurons[item + 1][0])}]" for item in range(len(neurons) - 2)]))
		if (len(neurons) - 2 > 0): print(f"Number of output neurons: {str(neurons[-1]) if (type(neurons[-1]) == int) else str(neurons[-1][0])}")
		if (len(normalization) > 0): print("Normalization size per layer:",  " | ".join([str(neurons[item + 1]) for item in range(len(normalization))]))
		if (len(normalization) > 0): print("Type of normalization per layer:", " | ".join([expand[normalization[item]] for item in range(len(normalization))]))

	elif (CNN != []): 

		print()
		print(f"Number of convolutional layers: {len(convolutions) - 1}")
		if (len(convolutions) - 1 > 0): print("Depth of each feature map:", " --> ".join([f"[{str(convolutions[item + 1])}]" for item in range(len(convolutions) - 1)]))
		if (len(kernel) > 0): print("Kernel window per layer:", " | ".join([f"{str(kernel[item])}x{str(kernel[item])}" if (type(kernel[item]) == int) else f"{str(kernel[item][0])}x{str(kernel[item][1])}" for item in range(len(kernel))]))
		if (len(stride) > 0): print("Stride pixels per layer:", " | ".join([f"{str(stride[item])}x{str(stride[item])}" if (type(stride[item]) == int) else f"{str(stride[item][0])}x{str(stride[item][1])}" for item in range(len(stride))]))
		if (len(padding) > 0): print("Padding width per layer:", " | ".join([f"{str(padding[item])}x{str(padding[item])}" if (type(padding[item]) == int) else f"{str(padding[item][0])}x{str(padding[item][1])}" for item in range(len(padding))]))
		if (len(normalization) > 0): print("Normalization size per layer:",  " | ".join([str(convolutions[item + 1]) for item in range(len(normalization))]))
		if (len(pooling) > 0): print("Pooling size per layer:", " | ".join([str(pooling[item][1]) for item in range(len(pooling))]))
		if (len(normalization) > 0): print("Type of normalization per layer:", " | ".join([expand[normalization[item]] for item in range(len(normalization))]))
		if (len(pooling) > 0): print("Type of pooling per layer:", " | ".join([expand[pooling[item][0]] for item in range(len(pooling))]))
		if (len(neurons) - 2 > 0): print(f"Number of hidden layers: {len(neurons) - 2}")
		if (len(neurons) - 2 > 0): print("Number of nodes per layer:", " --> ".join([f"[{str(neurons[item + 1]) if (type(neurons[item + 1]) == int) else str(neurons[item + 1][0])}]" for item in range(len(neurons) - 2)]))
		if (len(neurons) - 1 > 0): print(f"Number of output neurons: {str(neurons[-1]) if (type(neurons[-1]) == int) else str(neurons[-1][0])}")
		print(f"Input channel depth: {channels}")
		print(f"Input image size: {width}x{height}")

	elif (GAN != []):

		print()
		if (len(neurons) - 2 > 0): print(f"Number of generator hidden layers: {len(neurons[0]) - 2}")
		if (len(neurons) - 2 > 0): print(f"Number of critic hidden layers: {len(neurons[1]) - 2}")
		if (len(neurons) - 2 > 0): print("Number of generator nodes per layer:", " --> ".join([f"[{str(neurons[0][item + 1]) if (type(neurons[item + 1]) == int) else str(neurons[item + 1][0])}]" for item in range(len(neurons[0]) - 2)]))
		if (len(neurons) - 2 > 0): print("Number of critic nodes per layer:", " --> ".join([f"[{str(neurons[1][item + 1]) if (type(neurons[item + 1]) == int) else str(neurons[item + 1][0])}]" for item in range(len(neurons[1]) - 2)]))
		if (len(neurons) - 2 > 0): print(f"Number of generator output neurons: {str(neurons[0][-1]) if (type(neurons[0][0][-1]) == int) else str(neurons[0][-1][0])}")
		if (len(neurons) - 2 > 0): print(f"Number of critic output neurons: {str(neurons[1][-1]) if (type(neurons[1][1][-1]) == int) else str(neurons[1][-1][0])}")
		if (len(normalization[0]) > 0): print("Generator normalization size per layer:",  " | ".join([str(neurons[0][item + 1]) for item in range(len(normalization[0]))]))
		if (len(normalization[1]) > 0): print("Critic normalization size per layer:",  " | ".join([str(neurons[1][item + 1]) for item in range(len(normalization[1]))]))
		if (len(normalization[0]) > 0): print("Type of generator normalization per layer:", " | ".join([expand[normalization[0][item]] for item in range(len(normalization[0]))]))
		if (len(normalization[1]) > 0): print("Type of critic normalization per layer:", " | ".join([expand[normalization[1][item]] for item in range(len(normalization[1]))]))

	elif ((DCGAN != []) | (DCWGANGP != [])): 

		print()
		print(f"Number of generator convolutional layers: {len(convolutions[0]) - 1}")
		print(f"Number of critic convolutional layers: {len(convolutions[1]) - 1}")
		if (len(convolutions[0]) - 1 > 0): print("Depth of each generator feature map:", " --> ".join([f"[{str(convolutions[0][item + 1])}]" for item in range(len(convolutions[0]) - 1)]))
		if (len(convolutions[1]) - 1 > 0): print("Depth of each critic feature map:", " --> ".join([f"[{str(convolutions[1][item + 1])}]" for item in range(len(convolutions[1]) - 1)]))
		if (len(kernel[0]) > 0): print("Generator kernel window per layer:", " | ".join([f"{str(kernel[0][item])}x{str(kernel[0][item])}" if (type(kernel[0][item]) == int) else f"{str(kernel[0][item][0])}x{str(kernel[0][item][1])}" for item in range(len(kernel[0]))]))
		if (len(kernel[1]) > 0): print("Critic kernel window per layer:", " | ".join([f"{str(kernel[1][item])}x{str(kernel[1][item])}" if (type(kernel[1][item]) == int) else f"{str(kernel[1][item][0])}x{str(kernel[1][item][1])}" for item in range(len(kernel[1]))]))
		if (len(stride[0]) > 0): print("Generator stride pixels per layer:", " | ".join([f"{str(stride[0][item])}x{str(stride[0][item])}" if (type(stride[0][item]) == int) else f"{str(stride[0][item][0])}x{str(stride[0][item][1])}" for item in range(len(stride[0]))]))
		if (len(stride[1]) > 0): print("Critic stride pixels per layer:", " | ".join([f"{str(stride[1][item])}x{str(stride[1][item])}" if (type(stride[1][item]) == int) else f"{str(stride[1][item][0])}x{str(stride[1][item][1])}" for item in range(len(stride[1]))]))
		if (len(padding[0]) > 0): print("Generator padding width per layer:", " | ".join([f"{str(padding[0][item])}x{str(padding[0][item])}" if (type(padding[0][item]) == int) else f"{str(padding[0][item][0])}x{str(padding[0][item][1])}" for item in range(len(padding[0]))]))
		if (len(padding[1]) > 0): print("Critic padding width per layer:", " | ".join([f"{str(padding[1][item])}x{str(padding[1][item])}" if (type(padding[1][item]) == int) else f"{str(padding[1][item][0])}x{str(padding[1][item][1])}" for item in range(len(padding[1]))]))
		if (len(normalization[0]) > 0): print("Generator normalization size per layer:",  " | ".join([str(convolutions[0][item + 1]) for item in range(len(normalization[0]))]))
		if (len(normalization[1]) > 0): print("Critic normalization size per layer:",  " | ".join([str(convolutions[1][item + 1]) for item in range(len(normalization[1]))]))
		if (len(pooling[0]) > 0): print("Generator pooling size per layer:", " | ".join([str(pooling[0][item][1]) for item in range(len(pooling[0]))]))
		if (len(pooling[1]) > 0): print("Critic pooling size per layer:", " | ".join([str(pooling[1][item][1]) for item in range(len(pooling[1]))]))
		if (len(normalization[0]) > 0): print("Type of generator normalization per layer:", " | ".join([expand[normalization[0][item]] for item in range(len(normalization[0]))]))
		if (len(normalization[1]) > 0): print("Type of critic normalization per layer:", " | ".join([expand[normalization[1][item]] for item in range(len(normalization[1]))]))
		if (len(pooling[0]) > 0): print("Type of generator pooling per layer:", " | ".join([expand[pooling[0][item][0]] for item in range(len(pooling[0]))]))
		if (len(pooling[1]) > 0): print("Type of critic pooling per layer:", " | ".join([expand[pooling[1][item][0]] for item in range(len(pooling[1]))]))
		if (len(neurons[0]) - 2 > 0): print(f"Number of generator hidden layers: {len(neurons[0]) - 2}")
		if (len(neurons[1]) - 2 > 0): print(f"Number of critic hidden layers: {len(neurons[1]) - 2}")
		if (len(neurons[0]) - 2 > 0): print("Number of generator nodes per layer:", " --> ".join([f"[{str(neurons[0][item + 1]) if (type(neurons[0][item + 1]) == int) else str(neurons[0][item + 1][0])}]" for item in range(len(neurons[0]) - 2)]))
		if (len(neurons[1]) - 2 > 0): print("Number of critic nodes per layer:", " --> ".join([f"[{str(neurons[1][item + 1]) if (type(neurons[1][item + 1]) == int) else str(neurons[1][item + 1][0])}]" for item in range(len(neurons[1]) - 2)]))
		if (len(neurons[0]) - 1 > 0): print(f"Number of generator output neurons: {str(neurons[0][-1]) if (type(neurons[0][-1]) == int) else str(neurons[0][-1][0])}")
		if (len(neurons[1]) - 1 > 0): print(f"Number of critic output neurons: {str(neurons[1][-1]) if (type(neurons[1][-1]) == int) else str(neurons[1][-1][0])}")
		if (DCWGANGP != []): print(f"Number of critic iterations: {iterations}"), print(f"Penalty coefficient: {lamda}")
		print(f"Input channel depth: {channels}")
		print(f"Input image size: {width}x{height}")

	print(f"Learning rate: {('%f'%learning_rate).rstrip('0').rstrip('.')}")
	if (type(cost) == str): print(f"Cost function: {expand[cost.lower().lstrip().rstrip()]}") if (cost.lower().lstrip().rstrip() != "") else print(f"Cost function: custom")
	if (type(propagator) == str): print(f"Optimizer: {expand[propagator.lower().lstrip().rstrip()]}")
	elif ((type(propagator) != str) & (type(propagator[1]) == float)): print(f"Optimizer: {expand[propagator[0].lower().lstrip().rstrip()]} (momentum = {str(propagator[1])})")
	elif ((type(propagator) != str) & (type(propagator[1]) != float)): print(f"Optimizer: {expand[propagator[0].lower().lstrip().rstrip()]} (momentum = ({str(propagator[1][0])}, {str(propagator[1][1])}))")
	else: print(f"Optimizer: Adam")
	print(f"Batch size: {batch}")
	if (validator != None): print(f"Validation window: {horizon}")
	print(), print("*"*120)


def learn(file, learning_rate, episodes, cost, propagator, ANN = None, CNN = None, AE = None, train_percent = 0, validate_percent = 0, batch = 1, horizon = 0, minimum = None, maximum = None, rows = None, columns = None, directory = None, replacement = None, convert = None, show = True):

	os.system("cls")
	ANN, CNN, AE, rows, columns, replacement, convert = refresh(ANN, CNN, AE, rows, columns, replacement, convert)
	assert (ANN != []) | (CNN != []) | (AE != []), "A MODEL ARCHITECTURE IS REQUIRED"
	if (CNN != []): convolutions, kernel, stride, padding, normalization, pooling, functions, neurons, direction, position, offset = CNN
	elif (ANN != []): neurons, functions = ANN
	elif (AE != []): neurons, functions = AE
	collect, ratio, accuracy, residual, labels = [], [], [], [], []
	score, deviation, batch_accuracy, batch_error = [], [], [], []
	batch_residual, batch_score, correct, incorrect, flag = [], [], 0, 0, False
	neurons, functions, convolutions, error, dimension, size, width, height, channels, LUT, trainer, tester, validator, labels, mode, regression, flatten, unflatten = initialize_variables(file, cost, ANN, CNN, TNN, train_percent, validate_percent, batch, rows, columns, directory, replacement, minimum, maximum, convert, neurons, convolutions if (CNN != []) else None, functions)
	if (CNN != []): model = ConvolutionalNeuralNetwork(kernel, stride, padding, width, height, convolutions, functions, neurons, channels, pooling, direction, offset, normalization, position, flatten, unflatten).to(device)
	else: model = ArtificialNeuralNetwork(neurons, functions, flatten = flatten, unflatten = unflatten).to(device)

	if (type(propagator) == str): 

		if (propagator.lower().rstrip().lstrip() in optimization): optimizer = algorithm(model, optimization[propagator.lower().rstrip().lstrip()], learning_rate)
		else: optimizer = algorithm(model, optimization["adam"], learning_rate)

	elif (propagator[0].lower().rstrip().lstrip() in optimization): 

		if (propagator[0].lower().rstrip().lstrip() == "adam"): optimizer = algorithm(model, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, beta = propagator[1]) 
		elif ((propagator[0].lower().rstrip().lstrip() == "sgd") | (propagator[0].lower().rstrip().lstrip() == "rms")): optimizer = algorithm(model, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, momentum = propagator[1])
		elif (propagator[0].lower().rstrip().lstrip() in optimization): optimizer = algorithm(model, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate)
		else: optimizer = algorithm(model, optimization["adam"], learning_rate)

	information(ANN, CNN, [], [], [], [], model, learning_rate, cost, propagator, neurons, convolutions, kernel, stride, padding, pooling, normalization, width, height, channels, dimension, labels, 0, 0, validator, horizon, regression)
	model.train()

	for epoch in range(episodes):

		for index, (x, y) in enumerate(trainer):

			x, y = x.to(device), y.to(device)
			optimizer.zero_grad()
			prediction = model(x)
			loss = error(prediction, x if (AE != []) else y)
			loss.backward()
			optimizer.step()
			collect.append(loss.item())
			batch_error.append(loss.item())

			for cycle in range(len(prediction)):

				if (regression == True): 

					if (abs(prediction[cycle].item() - y[cycle].item()) < 0.8): correct += 1
					else: incorrect += 1

				elif (prediction.shape[-1] > 1):

					if (torch.argmax(prediction[cycle]) == y[cycle]): correct += 1
					else: incorrect += 1

				elif (prediction.shape[-1] == 1):

					if (min(labels, key = lambda x: abs(x - prediction[cycle].item())) == y[cycle].item()): correct += 1
					else: incorrect += 1

				total = correct + incorrect
				ratio.append(correct / total)
				batch_accuracy.append(correct / total)
				correct, incorrect = 0, 0

		residual.append(sum(collect) / len(collect))
		accuracy.append(sum(ratio) / len(ratio))
		if (validator != []): model, deviation, score, batch_residual, batch_score, flag = validate(model, validator, error, horizon, residual[-1], episodes, labels, True if (AE != []) else False, regression)
		collect, ratio = [], []
		correct, incorrect = 0, 0
		if ((show == False) & (flag == True)): break

		if (show == True):

			print("\nEpisode:", epoch + 1)
			if (validator == []): print("Error:", round(residual[-1], 4)), 
					      print("Accuracy:", round(accuracy[-1], 4))
			else: print("Training error:", round(residual[-1], 4)), 
			      print("Validation error:", round(deviation[-1], 4)), 
			      print("Training accuracy:", round(accuracy[-1], 4)), 
			      print("Validation accuracy:", round(score[-1], 4))
			if (flag == True): break

	if (tester != []): test(model, tester, regression)
	if ((LUT != []) & (CNN != [])): evaluate(model, height, LUT, channels)
	return model, residual, accuracy, batch_error, batch_accuracy, deviation, score, batch_residual, batch_score


def train(trainer, learning_rate, episodes, cost, propagator, GAN = None, DCGAN = None, DCWGANGP = None, flatten = None, unflatten = None, row = 0, column = 0, size = 0, name = None, show = True):

	os.system("cls")
	GAN, DCGAN, DCWGANGP, flatten, unflatten = refresh(GAN, DCGAN, DCWGANGP, flatten, unflatten)
	assert (GAN != []) | (DCGAN != []) | (DCWGANGP != []), "A MODEL ARCHITECTURE IS REQUIRED"
	if (DCGAN != []): convolutions, kernel, stride, padding, normalization, pooling, functions, direction, position, noise_width, channels, width, height, offset = DCGAN
	elif (DCWGANGP != []): convolutions, kernel, stride, padding, normalization, pooling, functions, direction, position, noise_width, length, lamda, channels, width, height, offset = DCWGANGP
	elif (GAN != []): neurons, functions, noise_width = GAN
	block_error, flag = [], False
	collect_generator, collect_critic, error_generator, error_critic = [], [], [], []
	decide_real, decide_fake, detect_real, detect_fake = [], [], [], []
	real_detection, fake_detection, generator_error, critic_error = [], [], [], []
	if ((type(cost) == str) & (GAN != [])): functions[0][-1], functions[1][-1] = "" if ((neurons[0][-1] > 1) & (cost.lower().rstrip().lstrip() == "ce") & ("softmax" in functions[0][-1])) else functions[0][-1], "" if ((neurons[1][-1] > 1) & (cost.lower().rstrip().lstrip() == "ce") & ("softmax" in functions[1][-1])) else functions[1][-1]
	if (GAN == []): generator = ConvolutionalNeuralNetwork(kernel[0], stride[0], padding[0], width, height, convolutions[0], functions[0], channels = channels, pooling = pooling, direction = direction[0], offset = offset[0], normalization = normalization[0], position = position[0], flatten = flatten[0], unflatten = unflatten[0]).to(device)
	else: generator = ArtificialNeuralNetwork(neurons[0], functions[0], unflatten = True).to(device)
	if (GAN == []): critic = ConvolutionalNeuralNetwork(kernel[1], stride[1], padding[1], width, height, convolutions[1], functions[1], channels = channels, pooling = pooling, direction = direction[1], offset = offset[1], normalization = normalization[1], position = position[1], flatten = flatten[1], unflatten = unflatten[1]).to(device) 
	else: critic = ArtificialNeuralNetwork(neurons[1], functions[1], flatten = True).to(device)

	if (type(propagator) == str): 

		if (propagator.lower().rstrip().lstrip() in optimization):

			optimizer_generator = algorithm(generator, optimization[propagator.lower().rstrip().lstrip()], learning_rate)  
			optimizer_critic = algorithm(critic, optimization[propagator.lower().rstrip().lstrip()], learning_rate)
	
		else: 

			optimizer_generator = algorithm(generator, optimization["adam"], learning_rate)
			optimizer_critic = algorithm(critic, optimization["adam"], learning_rate)

	elif (propagator[0].lower().rstrip().lstrip() in optimization): 
		
		if (propagator[0].lower().rstrip().lstrip() == "adam"):

			optimizer_generator = algorithm(generator, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, beta = propagator[1])  
			optimizer_critic = algorithm(critic, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, beta = propagator[1])

		elif ((propagator[0].lower().rstrip().lstrip() == "sgd") | (propagator[0].lower().rstrip().lstrip() == "rms")):  

			optimizer_generator = algorithm(generator, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, momentum = propagator[1]) 
			optimizer_critic = algorithm(critic, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, momentum = propagator[1]) 

		elif (propagator[0].lower().rstrip().lstrip() in optimization):

			optimizer_generator = algorithm(generator, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate) 
			optimizer_critic = algorithm(critic, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate)

		else:

			optimizer_generator = algorithm(generator, optimization["adam"], learning_rate)
			optimizer_critic = algorithm(critic, optimization["adam"], learning_rate)

	if callable(cost): error = cost 
	elif (type(cost) != str): error = criterion([utility[cost[0].lower().rstrip().lstrip() if (cost[0].lower().rstrip().lstrip() in utility) else "bce"], cost[1]]) 
	elif (cost.lower().rstrip().lstrip() in utility): error =  criterion(utility[cost.lower().rstrip().lstrip()])
	else: error = criterion(utility["bce"])	
	model = [generator, critic]
	dimension = [neurons[0][-1], neurons[1][-1]] if (GAN != []) else []
	information([], [], [], GAN, DCGAN, DCWGANGP, model, learning_rate, cost, propagator, neurons, convolutions, kernel, stride, padding, pooling, normalization, width, height, channels, dimension, labels, length, lamda, [], 0, False)
	fixed_noise = torch.randn(batch, noise_width, 1, 1).to(device)
	generator.train()
	critic.train()

	if (name != None):

		process = subprocess.Popen(f"tensorboard --logdir=runs --reload_multifile=true", shell = True)
		server = "http://localhost:6006/#images"
		writer_fake = SummaryWriter(f"runs/{name}/fake")
		writer_real = SummaryWriter(f"runs/{name}/real")
		writer_plot = SummaryWriter(f"runs/{name}/graphs")
		step, iteration = 0, 0
		settings = webdriver.ChromeOptions()
		settings.headless = False
		settings.add_experimental_option("excludeSwitches", ["enable-logging"])
		driver = webdriver.Chrome(executable_path = "chromedriver", options = settings)
		driver.get(server)
		driver.maximize_window()
		WebDriverWait(driver, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, ".mat-focus-indicator.mat-icon-button.mat-button-base"))).click()
		WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "mat-checkbox-1-input"))).click()
		webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()

	for epoch in range(episodes):

		for index, (x, y) in enumerate(trainer):

			x, y = x.to(device), y.to(device)
			label_real = torch.ones(len(x), 1).to(device)
			label_fake = torch.zeros(len(x), 1).to(device)

			if (length == 0):

				if (GAN == []): data_fake = torch.randn(len(x), noise_width, 1, 1).to(device)
				else: data_fake = torch.randn(len(x), noise_width).to(device)
				prediction_real = critic(x)
				loss_critic = error(prediction_real, label_real)
				image_fake = generator(data_fake)
				prediction_fake = critic(image_fake)
				loss_critic += error(prediction_fake, label_fake)
				optimizer_critic.zero_grad()
				loss_critic.backward()
				optimizer_critic.step()

			else: 

				for cycle in range(length):

					if (GAN == []): data_fake = torch.randn(len(x), noise_width, 1, 1).to(device)
					else: data_fake = torch.randn(len(x), noise_width).to(device)
					image_fake = generator(data_fake)
					prediction_real = critic(x)
					prediction_fake = critic(image_fake)
					penalty = gradient_penalty(critic, x, image_fake, device)
					loss_critic = -(torch.mean(prediction_real) - torch.mean(prediction_fake)) + lamda*penalty
					critic.zero_grad()
					loss_critic.backward(retain_graph = True)
					optimizer_critic.step()
					block_error.append(loss_critic.item())

			if (GAN == []): data_fake = torch.randn(len(x), noise_width, 1, 1).to(device)
			else: data_fake = torch.randn(len(x), noise_width).to(device)
			image_fake = generator(data_fake)
			prediction_fake = critic(image_fake)
			loss_generator = error(prediction_fake, label_real) if (length == 0) else -torch.mean(prediction_fake)
			optimizer_generator.zero_grad()
			loss_generator.backward() if (length == 0) else loss_generator.backward(retain_graph = True)
			optimizer_generator.step()
			decision_real = torch.mean((prediction_real > 0.5).float()).detach()
			decision_fake = torch.mean((prediction_fake > 0.5).float()).detach()
			collect_generator.append(loss_generator.item())
			collect_critic.append(loss_critic.item()) if (length == 0) else collect_critic.append(sum(block_error) / len(block_error))
			decide_real.append(decision_real.item())
			decide_fake.append(decision_fake.item())
			error_generator.append(loss_generator.item())
			detect_real.append(decision_real.item())
			detect_fake.append(decision_fake.item())
			block_error = []

			if (name != None):

				writer_plot.add_scalars("Error", { "Critic Error": collect_critic[-1], "Generator Error": collect_generator[-1] }, global_step = iteration)
				iteration += 1

				if (((index + 1)%period == 0) | (index  == 0)):

					with torch.no_grad():

						fake = generator(fixed_noise)
						img_grid_real = torchvision.utils.make_grid(real[:32], normalize = True)
						img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize = True)
						writer_real.add_image("Real", img_grid_real, global_step = step)
						writer_fake.add_image("Fake", img_grid_fake, global_step = step)

					step += 1

		generator_error.append(sum(collect_generator) / len(collect_generator))
		critic_error.append(sum(collect_critic) / len(collect_critic))
		real_detection.append(sum(decide_real) / len(decide_real))
		fake_detection.append(sum(decide_fake) / len(decide_fake))
		collect_generator, collect_critic = [], []
		decide_real, decide_fake = [], []

		if (show == True):

			print("\nEpisode:", epoch + 1)
			print("Generator error:", round(generator_error[-1], 4))
			print("Critic error:", round(critic_error[-1], 4))
			print("Real detection:", round(real_detection[-1], 4))
			print("Fake detection:", round(fake_detection[-1], 4))

	if (size != 0): processor(generator, discriminator, row, column, size)
    	if (name != None): process.send_signal(signal.SIGINT), driver.quit()
	return generator, critic, np.array(error_generator), np.array(error_critic), np.array(detect_real), np.array(detect_fake), np.array(generator_error), np.array(critic_error), np.array(real_detection), np.array(fake_detection)


def plot(data, colour, name, x, y, compare = False):

	plt.figure()
	axis = plt.axes(facecolor = "#E6E6E6")
	axis.set_axisbelow(True)
	plt.grid(color = "w", linestyle = "solid")
	for spine in axis.spines.values(): spine.set_visible(False)
	plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
	plt.tick_params(axis = "y", which = "both", left = False, right = False)

	if (compare == True): 

		if (type(colour[0]) != str): 

			if (type(name) != str):

				plt.plot(list(range(1, len(data[0]) + 1)), 
					 data[0], 
					 color = f"{colour[0][0] if (type(colour[0]) != str) else colour[0]}", 
					 alpha = colour[0][1] if (type(colour[0]) != str) else 1, 
					 linewidth = 1, 
					 label = name[0] if ((name[0] != "") & (name[0] != None) & (len(name) > 2)) else "_nolegend_")

		else: 

			plt.plot(list(range(1, len(data[0]) + 1)),
				      data[0], 
				      color = f"{colour[0][0] if (type(colour[0]) != str) else colour[0]}", 
				      alpha = colour[0][1] if (type(colour[0]) != str) else 1, 
				      linewidth = 1, 
				      label = "_nolegend_")

			if (type(name) != str): 

				plt.plot(list(range(1, len(data[1]) + 1)), 
					 data[1], 
					 color = f"{colour[1][0] if (type(colour[1]) != str) else colour[1]}", 
					 alpha = colour[1][1] if (type(colour[1]) != str) else 1, 
					 linewidth = 1, 
					 label = name[1] if ((name[1] != "") & (name[1] != None) & (len(name) > 2)) else "_nolegend_")

			else: 

				plt.plot(list(range(1, len(data[1]) + 1)), 
					 data[1], 
					 color = f"{colour[1][0] if (type(colour[1]) != str) else colour[1]}", 
					 alpha = colour[1][1] if (type(colour[1]) != str) else 1, 
					 linewidth = 1, 
					 label = "_nolegend_")

			if ((type(colour[0]) != str) | (type(colour[1]) != str)): 

				plt.scatter([list(range(1, len(data[1]) + 1))[-1]], 
					    [data[1][-1]], 
					    color = f"{colour[1][0] if (type(colour[1]) != str) else colour[1]}", 
					    marker = ".")

			if (type(name) != str): 

				plt.title(name[2] if (len(name) == 4) else name[0] if (len(name) == 2) else "")

	elif ((type(data[0]) != list) & (type(data[0]) != np.ndarray)): 

		plt.plot(list(range(1, len(data) + 1)), data, color = colour, linewidth = 1)
		if (type(name) == list): plt.title(name[0])
		else: plt.title("")

	else: 

		plt.scatter(data[0], data[1], color = colour, marker = ".", alpha = 0.4)

	plt.xlabel(x), plt.ylabel(y)
	if ((compare == True) & (type(name) != str)): plt.legend(loc = "best") if (((name[0] != "") & (name[0] != None) & (len(name) > 2)) | ((name[1] != "") & (name[1] != None) & (len(name) > 2))) else None
	plt.savefig(f"{name[-1] if (type(name) != str) else name}.png", dpi = 200)
	plt.show()


def test(model, data, regression):

	print()
	correct, incorrect, labels = 0, 0, []
	if (regression == False): labels += list(set([element.item() for element in list(itertools.chain.from_iterable([y for x, y in data]))]))
	model.eval()

	with torch.no_grad():

		for index, (x, y) in enumerate(data):

			prediction = model(x)

			for cycle in range(len(prediction)):

				if (prediction.shape[-1] > 1):

					if (torch.argmax(prediction[cycle]) == y[cycle]): correct += 1 
					else: incorrect += 1

				elif (prediction.shape[-1] == 1):

					if (regression == True):

						print(f"actual output: {round(y[cycle].item(), 2)} / predicted output: {round(prediction[cycle].item(), 2)} (residual = {round(abs(prediction[cycle].item() - y[cycle].item()), 4)})")

					else:

						if (min(labels, key = lambda x: abs(x - prediction[cycle].item())) == y[cycle].item()): correct += 1
						else: incorrect += 1

	total = correct + incorrect
	if ((incorrect == 1) & (regression == False)): print(f"\nCorrectly labeled {correct} samples and incorrectly labeled {incorrect} sample")
	elif ((correct == 1) & (regression == False)): print(f"\nCorrectly labeled {correct} sample and incorrectly labeled {incorrect} samples")
	elif (regression == False): print(f"\nCorrectly labeled {correct} samples and incorrectly labeled {incorrect} samples")
	if (regression == False): print(f"\nAccuracy: {round(100*(correct / total), 2)}%")


def evaluate(model, channel, width, height, LUT):

	black = (0, 0, 0)
	white = (255, 255, 255)
	size = (500, 500)
	click = False
	screen = pg.display.set_mode(size)
	screen.fill(white)
	pg.display.set_caption("Test")
	pg.init()

	while True: 

		for event in pg.event.get():

			if (event.type == pg.QUIT):

				pg.quit()
				sys.exit()

			if (event.type == pg.KEYDOWN):

				if (event.key == pg.K_ESCAPE): 

					screen.fill(white)

				elif (event.key == pg.K_SPACE): 

					matrix = np.transpose(np.array(pg.surfarray.pixels3d(screen)), axes = (1, 0, 2))
					data = torch.Tensor(normalize(matrix).reshape(1, channel, height, width))
					prediction = model(data)
					distribution = graph.softmax(prediction)
					category = torch.argmax(distribution).item()
					confidence = int(torch.max(distribution).item()*100)
					print(f"{str(LUT[category])} ({confidence}%)")
					pg.image.save(screen, f"{str(LUT[category])}_predicted.png")

			if (event.type == pg.MOUSEBUTTONDOWN):

				click = True

			elif (event.type == pg.MOUSEBUTTONUP):

				click = False

			elif ((event.type == pg.MOUSEMOTION) & (click == True)):

				x, y = pg.mouse.get_pos()
				location = (x, y)
				pg.draw.circle(screen, black, location, 10)

		pg.display.flip()


def validate(model, validator, error, horizon, residual, episodes, labels = None, mode = False, regression = False):
	
	if not all(hasattr(validate, item) for item in ["epoch", "validation_cost", "training_cost", "period", "increment", "history", "checkpoint", "cycle", "cost", "accuracy", "validation_error", "training_error", "batch_cost", "batch_accuracy"]):

		validate.epoch, validate.increment = 0, 0
		validate.cost, validate.accuracy = [], []
		validate.batch_cost, validate.batch_accuracy = [], []
		validate.validation_cost, validate.training_cost = [], []
		validate.validation_error, validate.training_error = 0, 0
		validate.period = horizon + validate.increment*(horizon + 1)
		validate.history = validate.checkpoint = validate.cycle = validate.period

	if ((validate.epoch == 0) | (validate.epoch == validate.history + 1)): _, validate.history = torch.save(model, "ANN.pth"), validate.period
	flag, correct, incorrect = False, 0, 0
	batch_error, batch_score = [], []
	labels = refresh(labels)
	count, counter = 0, 0
	model.eval()
	
	with torch.no_grad():

		for index, (x, y) in enumerate(validator):

			x, y = x.to(device), y.to(device)
			prediction = model(x)
			loss = error(prediction, y if (mode == False) else x)
			validate.batch_cost.append(loss.item())
			batch_error.append(loss.item())

			for iteration in range(len(prediction)):

				if (regression == True): 

					if (abs(prediction[iteration].item() - y[iteration].item()) < 0.8): correct += 1
					else: incorrect += 1

				elif (prediction.shape[-1] > 1):

					if (torch.argmax(prediction[iteration]) == y[iteration]): correct += 1
					else: incorrect += 1

				elif (prediction.shape[-1] == 1):

					if (min(labels, key = lambda x: abs(x - prediction[iteration].item())) == y[iteration].item()): correct += 1
					else: incorrect += 1

				total = correct + incorrect
				validate.batch_accuracy.append(correct / total)
				batch_score.append(correct / total)
				correct, incorrect = 0, 0

	model.train()
	validate.cost.append(sum(batch_error) / len(batch_error))
	validate.accuracy.append(sum(batch_score) / len(batch_score))
	if ((validate.epoch == 0) | (validate.epoch == validate.cycle + 1)): validate.validation_error, validate.training_error, validate.cycle = validate.cost[-1], residual, validate.period
	if (((validate.epoch > 0) & (validate.epoch <= horizon)) | ((validate.epoch >= validate.checkpoint + 2) & (validate.epoch <= validate.period))): validate.validation_cost.append(validate.cost[-1]), validate.training_cost.append(residual)
	if (validate.epoch == validate.period): validate.checkpoint = validate.period

	if (validate.epoch == validate.period):

		for item in range(horizon):

			if (round(validate.training_cost[item], 4) <= round(validate.training_error, 4)): counter += 1
			if (round(validate.validation_cost[item], 4) >= round(validate.validation_error, 4)): count += 1
			elif (round(validate.validation_cost[item], 4) < round(validate.validation_error, 4)): break

		if ((count == horizon) & (counter >= int(0.8*horizon))): 

			print(), print("*"*120)
			print(), print("TERMINATING LEARNING PROCESS")
			print(), print(f"\nTraining details: \n\ntruncated at {validate.epoch + 1} cycles out of {episodes}\n")
			print(), print(f"last discarded nodes;")

			for tag, parameter in model.named_parameters(): 

				try: print(f"\n\n{tag}:\n", parameter.detach().numpy(), "\n\n")	
				except Exception as error: print(error)

			if (len([item for item in validate.validation_cost if (round(item, 4) == round(validate.validation_error, 4))]) != horizon): model = torch.load("ANN.pth")
			elif (len([item for item in validate.validation_cost if (round(item, 4) == round(validate.validation_error, 4))]) == horizon): _, model = torch.save(model, "ANN.pth"), torch.load("ANN.pth")
			print(), print(f"last reverted nodes;")

			for tag, parameter in model.named_parameters():

				try: print(f"\n\n{tag}:\n", parameter.detach().numpy(), "\n\n")	
				except Exception as e: print(e)

			print(), print("*"*120)
			flag = True

		validate.increment += 1
		validate.period = horizon + validate.increment*(horizon + 1)
		validate.validation_cost, validate.training_cost = [], []

	validate.epoch += 1
	return model, validate.cost, validate.accuracy, validate.batch_cost, validate.batch_accuracy, flag


def processor(generator, critic, row, column, size, noise_width, name, flag = False):

	print()
	generator.eval(), critic.eval()
	if (flag == True): image = generator(torch.randn(size, noise_width, 1, 1).to(device))
	else: image = generator(torch.randn(size, noise_width).to(device))
	figure, axes = plt.subplots(row, column, figsize = (2*column, 2*row))
	expand = list(image.size())
	expand[0] = 1

	for index, axis in enumerate(axes.flatten()):

		axis.imshow(image[index, :, ].detach().cpu().squeeze(), cmap = "gray")
		print(f"image {index + 1}:", "Real" if (critic(image[index, :, ].reshape(expand)).item() > 0.5) else "Fake")
		axis.axis("off")

	plt.savefig(f"{name}.png", dpi = 200)
	plt.show()


def gradient_penalty(critic, real_image, fake_image):

	batch, channel, height, width = real_image.shape
	epsilon = torch.rand((batch, 1, 1, 1)).repeat(1, channel, height, width).to(device)
	interpolation = real_image*epsilon + fake_image*(1 - epsilon)
	score = critic(interpolation)
	gradient = torch.autograd.grad(inputs = interpolation, outputs = score, grad_outputs = torch.ones_like(score), create_graph = True, retain_graph = True)[0]
	gradient = gradient.view(gradient.shape[0], -1)
	average = gradient.norm(2, dim = 1)
	penalty = torch.mean((average - 1)**2)
	return penalty


def savitzky_golay(data, window, order, derivative = 0, rate = 1):

	try: window, order = np.abs(np.int(window)), np.abs(np.int(order))
	except ValueError: raise ValueError("WINDOW AND ORDER MUST BE A POSITIVE INTEGER")
	if ((window%2 != 1) | window) < 1: raise TypeError("WINDOW MUST BE A POSITIVE ODD INTEGER")
	if (window < order + 2): raise TypeError("WINDOW IS TOO SMALL FOR THE POLYNOMIAL ORDER")
	if (type(data) != np.ndarray): data = np.asarray(data)
	order_range = range(order + 1)
	half_window = (window - 1) // 2
	offset = np.mat([[k**index for index in order_range] for k in range(-half_window, half_window + 1)])
	gradient = np.linalg.pinv(offset).A[derivative]*factorial(derivative)*rate**derivative
	first = data[0] - np.abs(data[1:half_window + 1][::-1] - data[0] )
	last = data[-1] + np.abs(data[-half_window - 1:-1][::-1] - data[-1])
	data = np.concatenate((first, data, last))
	filtered = np.convolve(gradient[::-1], data, mode = "valid")
	return filtered


def truncate(data, rows = None, columns = None, field = None):

	rows, columns = refresh(rows, columns)

	if (columns != []):

		if (type(columns) == tuple):

			if (type(columns[0]) == int):

				for item in range(sorted(columns)[1] - 1, sorted(columns)[0] - 2, -1): 

					data.drop(data.columns[item], axis = 1, inplace = True)

			elif (type(columns[0]) == str):

				category = list(data.columns)
				category = (category.index(columns[0]) + 1, category.index(columns[1]) + 1)

				for item in range(sorted(category)[1] - 1, sorted(category)[0] - 2, -1): 

					data.drop(data.columns[item], axis = 1, inplace = True)

		elif (type(columns) == list):

			if (type(columns[0]) == str): 

				for item in columns: 

					data = data.drop(item, axis = 1)

			elif (type(columns[0]) == int):

				for item in sorted(columns, reverse = True): 

					data.drop(data.columns[item - 1], axis = 1, inplace = True)

	if (rows != []):

		for row in rows:

			if (row not in data[list(data.columns)[-1]]):

				data = data[~data.eq(row).any(1)]

			else:

				category = list(data.columns)
				field = category[-1]
				if (type(field) == str): data = data.loc[data[field] != row] if (field.isdigit() == False) else data.loc[data.iloc[:, -1] != row]
				else: data = data.loc[data.iloc[:, -1] != row]

	return data


def one_hot_encoding(data):

	sections = list(set([item if (type(item) != np.ndarray) else item[0] for item in data]))
	LUT = {sections[item]: item if (type(item) != np.ndarray) else item[0] for item in range(len(sections))}
	update = np.zeros((len(data), len(sections)))

	for index in range(len(data)):

		encoder = np.zeros(len(sections))
		encoder[LUT[data[index]] if (type(data[index]) != np.ndarray) else data[index][0]] = 1
		update[index] = encoder

	return update


def replace(data, replacer):

	for index in range(len(replacer)):

		replacement = {replacer[index][item][0]: replacer[index][item][1] for item in range(len(replacer[index]) - 1)}
		if (type(replacer[index][-1]) == str): data[replacer[index][-1] - 1] = data[replacer[index][-1] - 1].apply(lambda item: replacement[item])
		else: data.iloc[:, replacer[index][-1] - 1] = data.iloc[:, replacer[index][-1] - 1].apply(lambda item: replacement[item])

	return data


def initialize_variables(file, cost, ANN, CNN, AE, GAN, DCGAN, DCWGANGP, train_percent, validate_percent, batch, rows, columns, directory, replacement, minimum, maximum, convert, neurons = None, convolutions = None, functions = None):

	neurons, convolutions, functions = refresh(neurons, convolutions, functions)

	if ((ANN != []) | (CNN != [])):

		collection = set()
		mode, img = False, False
		count, counter, header = 0, 0, 0
		trainer, tester, validator = [], [], []
		X, Y, labels, replacer, dataset = [], [], [], [], []
		if ((ANN != []) & (len(functions) == 0) & (cost.lower().lstrip().rstrip() == "bcel") & (len(neurons) == 0)): functions = [""]
		elif ((ANN != []) & (len(functions) == 0) & (len(neurons) == 0)): functions = ["sigmoid"]

		try:

			data = pd.read_csv(file if (type(file) == str) else file[-1], header = 0)

		except Exception as error:
			
			print(), print("Error:", error)
			X, y = extract(file, directory, convert, flag = None)
			if (len(X.shape) <= 2): data = pd.DataFrame(torch.cat([X, y], dim = 1).numpy())
			else: categories, img = y, True

		headings = list(data.columns) if (img == False) else []
		length = len(functions) + 1 if (ANN != []) else len(functions) - len(convolutions)
		categories = data.iloc[:, -1].values if (img == False) else categories.tolist()
		classes = [element for element in categories if not (element in collection or collection.add(element))]
		catalogue = collections.Counter(categories)
		labels = [item for item in catalogue]
		decimal, integer, regression = 0, 0, None
		
		for item in labels:

			if ((type(item) != str) | ((type(item) == str) & (defloat(item).replace(".", "").isdigit() == True))):

				if (float(item) == int(float(item))): integer += 1
				else: decimal += 1

			elif (defloat(item).replace(".", "").isdigit() == False):

				regression = False
				break

		if (regression == None):

			regression = True if ((decimal > 0) | (len(labels) > len(categories)*0.1)) else False

		unique = [item for item in catalogue if (catalogue[item] > 1)]
		output = len(unique)
		properties = len(data.columns) - 1 if (img == False) else 0
		if (len(neurons) > 0): sections = output if (((len(neurons) == length - 2) & (ANN != [])) | ((len(neurons) == length - 1) & (type(neurons[0]) == str)) | ((len(neurons) == length - 1) & (type(neurons[-1]) == str)) | ((len(neurons) == length) & (type(neurons[-1]) == str))) else neurons[-1] if (type(neurons[-1]) == int) else neurons[-1][0]
		else: sections = 1
		if (img == True): rows = []
		else: rows.extend(["?", "nan"])

		for heading in headings:

			if (type(heading) == str): count, counter = count + 1 if (heading.replace(".", "").isdigit() == False) else count, counter + 1 if (heading.replace(".", "").isdigit() == True) else counter
			elif (type(heading) != str): counter += 1
			if (counter > 0): break

		if ((count == len(headings)) & (img == False)): ticket = headings[-1]
		elif (img == False): ticket = None
		if (regression == False): encoding = [(classes[index], index) for index in range(len(classes))] if (sections > 1) else [(classes[index], value[functions[-1]] + index / (len(classes) - 1)) for index in range(len(classes))]
		else: encoding = []

		for element in range(properties):

			inventory = collections.Counter(data.iloc[:, element]) if (str(headings[-1]).replace(".", "").isdigit() == True) else collections.Counter(data.iloc[1:, element])

			for item in inventory:

				if ((not defloat(item).replace(".", "").isdigit()) & (item not in ["?", "nan"])):

					replacer.append((item, header))
					header += 1

			if (len(replacer) != 0): replacer.append(element + 1), replacement.append(replacer)
			replacer, header = [], 0

		encoding = list(encoding)
		encoding.append(replacement)
		if (regression == False): rows = rows + [item for item in catalogue if (catalogue[item] <= 1)]

		if ((train_percent > 0) & (type(file) == str) & (X == [])):

			X, y = extract(file, directory, encoding, convert, rows, columns, sections, ticket, flag = None)

		elif ((train_percent == 0) & (X == [])):

			for index in range(3):

				item = "training" (if index == 0) else "testing" (if index == 1) else "validation"
				family = "train" (if index == 0) else "test" (if index == 1) else "validate"
				suffix = "_trainer"  (if index == 0) else "_tester" (if index == 1) else "_validator"
				name = file[index].lower().replace(item, "").replace(family, "").replace("dataset", "").replace("data", "").replace("_", "").replace(".", "").replace("csv", "").replace("txt", "").lstrip().rstrip().replace(" ", "_") + suffix

				if f"{name}.pth" not in os.listdir(os.getcwd()):

					if (index == 0): 

						train = ImageData(f"{file[index]}", directory[index], transformation.Compose(convert), True, False)
						current = trainer = loader(dataset = train, batch_size = batch, shuffle = True, drop_last = False, num_workers = 2)

					elif (index == 1): 

						test = ImageData(f"{file[index]}", directory[index], transformation.Compose(convert), True, False)
						current = tester = loader(dataset = test, batch_size = batch, shuffle = True, drop_last = False, num_workers = 2)

					else: 

						validate = ImageData(f"{file[index]}", directory[index], transformation.Compose(convert), True, False)
						current = validator = loader(dataset = validate, batch_size = batch, shuffle = True, drop_last = False, num_workers = 2)

					torch.save(current, f"{name}.pth")
		
				else:

					if (index == 0): trainer = torch.load(f"{name}.pth")
					elif (index == 1): tester = torch.load(f"{name}.pth")
					else: validator = torch.load(f"{name}.pth")

			X, y = iter(trainer).next()
			Y = [element.item() for element in y]

		dimension, size = len(X[0].flatten() if (len(X.shape) > 2) else X[0]), len(collections.Counter(y.numpy().reshape((len(y), )) if (type(y) == torch.Tensor) else y.reshape((len(y), )))) if ((regression == False) & (len(Y) == 0)) else len(list(set(Y))) if (len(Y) > 0) else sections
		width, height, channels = X[1].shape[-1] if (CNN != []) else 0, X[0].shape[-1] if (CNN != []) else 0, X[0].shape[-3] if ((CNN != []) | ((ANN != []) & (len(X.shape) > 2))) else 0
		flatten, unflatten = True if ((CNN != []) | ((ANN != []) & (len(X.shape) > 2))) else False, False
		if ((minimum != None) & ((maximum != None))): X = scale(X, minimum, maximum)
		if ((trainer == []) & (tester == []) & (validator == [])): trainer, tester, validator = partition(X, y, size, batch, train_percent, validate_percent, False if (ANN != []) else True)
		LUT = {index: unique[index] for index in range(len(unique))} if ((CNN != []) & (regression == False)) else []
		if (type(cost) == str): functions[-1] = "" if (((size > 1) & (cost.lower().rstrip().lstrip() == "ce") & ("softmax" in functions[-1])) | ((size == 1) & (cost.lower().rstrip().lstrip() == "bcel") & (functions[-1] == "sigmoid"))) else functions[-1]
		if (type(cost) == str): cost, functions[-1] = "ce" if ((sections > 1) & (cost.lower().lstrip().rstrip() != "ce")) else cost, "" if ((sections > 1) & (cost.lower().lstrip().rstrip() != "ce")) else functions[-1]
		if (CNN != []): convolutions.insert(0, channels)

		if ((len(neurons) == length - 2) & (ANN != [])):

			neurons.insert(0, dimension)
			neurons.append(size if (regression == False) else 1)

		elif (len(neurons) == length - 1):

			if ((type(neurons[0]) == str) & (ANN != [])):

				neurons[0] = (dimension, neurons[0])
				neurons.append(size if (regression == False) else 1)

			elif ((type(neurons[-1]) == str) & (ANN != [])):

				neurons[-1] = (size if (regression == False) else 1, neurons[-1])
				neurons.insert(0, dimension)

			elif ((type(neurons[0]) != str) & (ANN != [])):

				neurons.insert(0, dimension)

			elif (CNN != []): 

				neurons.append(size if (regression == False) else 1)

		elif (len(neurons) == length):

			if (type(neurons[-1]) == str):

				if (ANN != []): neurons[0] = (dimension, neurons[0])
				neurons[-1] = (size if (regression == False) else 1, neurons[-1])

			elif ((type(neurons[-1]) != str) & (ANN != [])):

				neurons[0] = (dimension, neurons[0])

		elif ((len(neurons) == 0) & (ANN != [])):

			neurons.insert(0, 1)
			neurons.append(dimension)

	if (callable(cost) == True): error = cost
	elif (type(cost) != str): error = criterion([utility[cost[0].lower().rstrip().lstrip() if (cost[0].lower().rstrip().lstrip() in utility) else "ce" if (size > 1) else "mse"], cost[1]]) 
	elif (cost.lower().rstrip().lstrip() in utility): error = criterion(utility[cost.lower().rstrip().lstrip()])
	elif (type(cost) == str): error = criterion(utility["ce" if (size > 1) else "mse"])
	else: error = criterion(utility["mse"])
	return neurons, functions, convolutions, error, dimension, size, width, height, channels, LUT, trainer, tester, validator, labels, mode, regression, flatten, unflatten


def defloat(value):

	try: value = ("%.20f"%abs(float(value))).rstrip("0").rstrip(".")
	except: pass
	return str(value)
