from PIL import Image
import pandas as pd
import numpy as np
import spacy, itertools, torch, cv2, imutils
import torch.nn as NN
import torch.optim as solver
import torch, torchvision, os
from torch.utils.data import DataLoader as loader
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset as group
from sklearn.model_selection import train_test_split as split
import torchvision.transforms as transformation
import torch.nn.functional as graph
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from torchsummary import summary
from torchtext.data.metrics import bleu_score
plt.rcParams["font.family"] = "Arial"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


activation = {"relu": "ReLU", "tanh": "Tanh", "sigmoid": "Sigmoid", "softmax": "Softmax", "logsoftmax": "LogSoftmax", "leaky": "LeakyReLU"}
utility = {"nll": "NLLLoss", "bce": "BCELoss", "mse": "MSELoss", "crossentropy": "CrossEntropyLoss"}
optimization = {"adam": "Adam", "rms": "RMSProp", "sgd": "SGD"}
dataset = {"mnist": "MNIST", "cifar": "CIFAR10", "celeb": "CelebA", "fashion": "FashionMNIST", "emnist": "EMNIST"}
normalizer = {"batch": "BatchNorm2d", "instance": "InstanceNorm2d", "layer": "LayerNorm2d"}
pool = {"average": "AvgPool2d", "max": "MaxPool2d"}
language = {"english": "en_core_web_sm", "german" : "de_core_news_sm"}
function = lambda transform, slope = None: getattr(torch.nn, transform)(dim = 1) if ("Softmax" in transform) else getattr(torch.nn, transform)(slope) if (("LeakyReLU" in transform) & (slope != None)) else getattr(torch.nn, transform)()
criterion = lambda score: getattr(torch.nn, score[0])(ignore_index = score[1]) if (type(score) != str) else getattr(torch.nn, score)()
algorithm = lambda model, method, learning_rate, momentum = 0, beta = (): getattr(torch.optim, method)(model.parameters(), lr = learning_rate, betas = beta) if (beta != ()) else getattr(torch.optim, method)(model.parameters(), lr = learning_rate, momentum = momentum) if (momentum != 0) else getattr(torch.optim, method)(model.parameters(), lr = learning_rate)
library = lambda family, folder, convert: getattr(torchvision.datasets, family)(root = folder, transform = convert, download = True)
normalize = lambda data: (data - data.min()) / (data.max() - data.min())
aggregate = lambda transform, dimension: getattr(torch.nn, transform)(dimension, affine = True) if ("Instance" in transform) else getattr(torch.nn, transform)(dimension)
sampling = lambda transform, window: getattr(torch.nn, transform)(window)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArtificialNeuralNetwork(NN.Module):

	forward = lambda self, data: self.network(data)

	def __init__(self, nodes, functions, system = None, flatten = False, unflatten = False, channels = 1, height = 28):

		super().__init__()
		depth = len(nodes) - 1
		self.network = system if (system != None) else NN.Sequential()

		for index in range(depth):

			if ((index == 0) & (flatten == True)): self.network.add_module("transform", NN.Flatten())
			self.network.add_module(f"layer {index + 1}", NN.Linear(nodes[index], nodes[index + 1]))
			if (type(functions[index]) != str): self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()])) if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 2) & (functions[index][-1] >= 1)) else self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()], functions[index][1])) if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 2) & (functions[index][-1] < 1)) else self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()], functions[index][1])) if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 3)) else self.network.add_module(f"activity {index + 1}", function(activation["tanh"])), self.network.add_module(f"drop {index + 1}", torch.nn.Dropout(functions[index][-1] / 100)) if (functions[index][-1] >= 1) else None
			elif (functions[index].lower().rstrip().lstrip() in activation): self.network.add_module(f"activity {index + 1}", function(activation[functions[index].lower().rstrip().lstrip()]))
			elif (functions[index].lower().rstrip().lstrip() != ""): self.network.add_module(f"activity {index + 1}", function(activation["tanh"]))
			if ((index == depth - 1) & (unflatten == True)): self.network.add_module("transform", NN.Unflatten(1, (channels, height, height)))


class ConvolutionalNeuralNetwork(NN.Module):

	dimension = lambda self, pixels, kernel, padding = 1, stride = 1: np.floor((pixels + 2*padding - kernel) / stride) + 1
	forward = lambda self, data: self.network(data)

	def __init__(self, kernel, stride, padding, height, convolutions, functions, nodes = [], channels = 1, pooling = [], direction = 1, offset = True, normalization = [], flatten = False, unflatten = False):

		super().__init__()
		depth = len(convolutions) - 1
		self.network = NN.Sequential()
		if (normalization == []): normalization = [("", 0)]*depth
		if (pooling == []): pooling = [("", 0)]*depth
		
		for index in range(depth):

			if (direction == -1): self.network.add_module(f"transpose convolution {index + 1}", NN.ConvTranspose2d(convolutions[index], convolutions[index + 1], kernel[index], stride[index], padding[index], bias = offset))
			else: self.network.add_module(f"convolution {index + 1}", NN.Conv2d(convolutions[index], convolutions[index + 1], kernel[index], stride[index], padding[index], bias = offset))
			if (pooling[index][0] != ""): self.network.add_module(f"pool {index + 1}", sampling(pool[pooling[index][0]], pooling[index][1])) if (pooling[index][0].lower().lstrip().rstrip() in pool) else self.network.add_module(f"pool {index + 1}", sampling(pool["max"], pooling[index][1]))
			if ((normalization[index][0] != "") & (normalization[index][-1] == 0)): self.network.add_module(f"norm {index + 1}", aggregate(normalizer[normalization[index][0]], normalization[index][1])) if (normalization[index][0].lower().lstrip().rstrip() in normalizer) else self.network.add_module(f"norm {index + 1}", aggregate(normalizer["batch"], normalization[index][1]))
			if (type(functions[index]) != str): self.network.add_module(f"activation {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()])) if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 2) & (functions[index][-1] >= 1)) else self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()], functions[index][1])) if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 2) & (functions[index][-1] < 1)) else self.network.add_module(f"activity {index + 1}", function(activation[functions[index][0].lower().rstrip().lstrip()], functions[index][1])) if ((functions[index][0].lower().rstrip().lstrip() in activation) & (functions[index][0].lower().rstrip().lstrip() != "") & (len(functions[index]) == 3)) else self.network.add_module(f"activity {index + 1}", function(activation["relu"])), self.network.add_module(f"drop {index + 1}", torch.nn.Dropout(functions[index][-1] / 100)) if (functions[index][-1] >= 1) else None
			elif (functions[index].lower().rstrip().lstrip() in activation): self.network.add_module(f"activation {index + 1}", function(activation[functions[index].lower().rstrip().lstrip()]))
			elif (functions[index].lower().rstrip().lstrip() != ""): self.network.add_module(f"activation {index + 1}", function(activation["relu"]))
			if ((normalization[index][0] != "") & (normalization[index][-1] != 0)): self.network.add_module(f"norm {index + 1}", aggregate(normalizer[normalization[index][0]], normalization[index][1])) if (normalization[index][0].lower().lstrip().rstrip() in normalizer) else self.network.add_module(f"norm {index + 1}", aggregate(normalizer["batch"], normalization[index][1]))
			if ((index == 0) & (pooling[index][0] != "")): size = self.dimension(height, kernel[index], padding[index], stride[index]) // pooling[index][1]
			elif (pooling[index][0] != ""): size = self.dimension(size, kernel[index], padding[index], stride[index]) // pooling[index][1]
			if ((nodes == []) & (flatten == True) & (index == depth - 1)): self.network.add_module("transform", NN.Flatten())
			if ((index == depth - 1) & (nodes != []) & (pooling[index][0] != "")): nodes.insert(0, convolutions[-1]*int(self.dimension(size, 1, 0, 1)*self.dimension(size, 1, 0, 1))), ArtificialNeuralNetwork(nodes, functions[depth:], self.network, flatten, unflatten, channels, height)


class TransformerNeuralNetwork(NN.Module):

	mask = lambda self, source: source.transpose(0, 1) == self.padding_index

	def __init__(self, width_embedding, width_source_vocabulary, width_target_vocabulary, width_encoder, width_decoder, padding_index, heads, expansion, maximum, drop_percent = 0):

		super().__init__()
		self.padding_index = padding_index
		self.source_word_embedding = NN.Embedding(width_source_vocabulary, width_embedding)
		self.source_position_embedding = NN.Embedding(maximum, width_embedding)
		self.target_word_embedding = NN.Embedding(width_target_vocabulary, width_embedding)
		self.target_position_embedding = NN.Embedding(maximum, width_embedding)
		self.transformer = NN.Transformer(width_embedding, heads, width_encoder, width_decoder, expansion, drop_percent)
		self.dropout = NN.Dropout(drop_percent)
		self.ANN = NN.Linear(width_embedding, width_target_vocabulary)

	def forward(self, source, target):

		source_padding_mask = self.mask(source)
		length_source, width_source = source.shape
		length_target, width_target = target.shapes
		position_source = (torch.arange(0, length_source).unsqueeze(1).expand(length_source, width_source).to(device))
		position_target = (torch.arange(0, length_target).unsqueeze(1).expand(length_target, width_target).to(device))
		embed_source = self.dropout(self.source_word_embedding(source) + self.source_position_embedding(position_source))
		embed_target = self.dropout(self.target_word_embedding(target) + self.target_position_embedding(position_target))
		target_mask = self.transformer.generate_square_subsequent_mask(length_target).to(device)
		output = self.transformer(embed_source, embed_target, src_key_padding_mask = source_padding_mask, tgt_mask = target_mask)
		output = self.ANN(output)
		return output




class ImageData(Dataset):

	__len__ = lambda self: self.y.shape[0]

	def __init__(self, csv_folder, image_folder, transform = None, grey = False, flag = False, row = None):

		df = pd.read_csv(csv_folder, header = row)
		self.image_folder = image_folder
		self.image_labels = df.iloc[:, 0].values
		if (flag == False): self.y = df.iloc[:, 1].values
		else: self.y = data.iloc[:, -1:].values
		self.transform = transform
		self.grey = grey

	def __getitem__(self, index):

		if (self.grey == True): image = Image.open(os.path.join(self.image_folder, self.image_labels[index])).convert('L')
		else: image = Image.open(os.path.join(self.image_folder, self.image_labels[index]))
		if self.transform is not None: image = self.transform(image)
		label = self.y[index]
		return image, label


def scale(data, minimum, maximum): 

	high, low = max(data) if (type(data) != torch.Tensor) else data.max().item(), min(data) if (type(data) != torch.Tensor) else data.min().item()
	gradient = fsolve(lambda slope, data, minimum, maximum: ((maximum - minimum + slope*low) / high) - slope, 1, args = (data, minimum, maximum))[0]
	offset = minimum - gradient*low
	data = gradient*data + offset
	return data


def extract(file, directory = None, encoding = None, convert = [], output = None, label = None, flag = None):

	if (directory == None):

		ticket = 0 if (label != None) else None
		data = pd.read_csv(f"{file}", header = ticket)
		dimension = len(data.iloc[0]) - 1
		mapping = {encoding[index][0]: encoding[index][1] for index in range(len(encoding))}
		if (label != None): data[f"{label}"] = data[f"{label}"].apply(lambda index: mapping[index])
		else: data.iloc[:, -1] = data.iloc[:, -1].apply(lambda index: mapping[index])
		x = normalize(data.iloc[:, 0:dimension].values)
		if (output > 1): y = data.iloc[:, -1].values
		else: y = data.iloc[:, -1:].values
		if (flag != None): y = normalize(y)

	else:

		X, Y = [], []
		convert.insert(0, transformation.ToTensor())
		data = library(dataset[file.lower().rstrip().lstrip()], directory, transformation.Compose(convert)) if (file.lower().rstrip().lstrip() in dataset) else library(dataset["mnist"], directory, transformation.Compose(convert))
		Z = loader(dataset = data, batch_size = 1, num_workers = 2)
		for x, y in Z: X.append(x), Y.append(y)
		x, y = torch.cat(X, dim = 0), torch.cat(Y, dim = 0)
		if ((torch.max(x) > 255) | (torch.min(x) < 0)): x = normalize(x)
		if (flag != None): y = normalize(y)

	return x, y


def partition(characteristics, categories, output, batch, training_percentage = 100, validation_percentage = 0, flag = False):

	train_percent = training_percentage / 100
	validate_percent = validation_percentage / 100
	test_percent = 1 - (train_percent + validate_percent)
	if ((output == 1) & (isinstance(categories[0], np.ndarray) == False) & (flag == False)): categories = categories.reshape((len(categories), -1))
	training_x, training_y = characteristics, categories
	train_x, train_y = torch.FloatTensor(training_x), training_y
	if (output > 1): train_y = torch.LongTensor(training_y)
	elif (output == 1): train_y = training_y.float()
	train = group(train_x, train_y)
	trainer = loader(dataset = train, batch_size = batch, shuffle = True, num_workers = 2)
	tester, validater = None, None

	if (test_percent > 0):

		training_x, testing_x, training_y, testing_y = split(characteristics, categories, test_size = test_percent, random_state = np.random.randint(1, 100), shuffle = True, stratify = categories)
		train_x = torch.FloatTensor(training_x)
		if (output > 1): train_y = torch.LongTensor(training_y)
		else: train_y = training_y.float()
		train = group(train_x, train_y)
		trainer = loader(dataset = train, batch_size = batch, shuffle = True, num_workers = 2)
		test_x = torch.FloatTensor(testing_x)
		if (output > 1): test_y = torch.LongTensor(testing_y)
		else: test_y = testing_y.float()
		test = group(test_x, test_y)
		tester = loader(dataset = test, batch_size = batch, shuffle = True, num_workers = 2)

	if (validate_percent > 0):

		training_x, validation_x, training_y, validation_y = split(training_x, training_y, test_size = validate_percent, random_state = np.random.randint(1, 100), shuffle = True, stratify = training_y)
		train_x = torch.FloatTensor(training_x)
		if (output > 1): train_y = torch.LongTensor(training_y)
		else: train_y = training_y.float()
		train = group(train_x, train_y)
		trainer = loader(dataset = train, batch_size = batch, shuffle = True, num_workers = 2)
		validate_x = torch.FloatTensor(validation_x)
		if (output > 1): validate_y = torch.LongTensor(validation_y)
		else: validate_y = validation_y.float()
		validate = group(validate_x, validate_y)
		validater = loader(dataset = validate, batch_size = batch, shuffle = True, num_workers = 2)

	return trainer, tester, validater


def learn(trainer, functions, learning_rate, episodes, cost, propagator, ANN = [], CNN = [], TNN = [], validater = [], horizon = 0, flatten = False, unflatten = False, show = True):

	assert (ANN != []) | (CNN != []) | (TNN != []), "A MODEL ARCHITECTURE IS REQUIRED"
	if (CNN != []): neurons, kernel, stride, padding, channels, height, pooling, convolutions, direction, offset, normalization = CNN
	elif (TNN != []): neurons, width_embedding, width_source_vocabulary, width_target_vocabulary, padding_index, heads, width_encoder, width_decoder, expansion, dropout_percent, maximum = TNN
	elif (ANN != []): neurons, channels, height = ANN
	collect, ratio = [], []
	accuracy, residual = [], []
	score, deviation = [], []
	batch_accuracy, batch_error = [], []
	correct, incorrect = 0, 0
	Y, labels, flag = [], [], False
	if (type(cost) == str): functions[-1] = "" if ((neurons[-1] > 1) & (cost.lower().rstrip().lstrip() == "crossentropy") & ("softmax" in functions[-1])) else functions[-1]
	model = ConvolutionalNeuralNetwork(kernel, stride, padding, height, convolutions, functions, neurons, channels, pooling, direction, offset, normalization, flatten, unflatten).to(device) if (CNN != []) else TransformerNeuralNetwork(width_embedding, width_source_vocabulary, width_target_vocabulary, padding_index, heads, width_encoder, width_decoder, expansion, dropout_percent, maximum).to(device) if (TNN != []) else ArtificialNeuralNetwork(neurons, functions, None, flatten, unflatten, channels, height).to(device)
	if (type(propagator) == str): optimizer = algorithm(model, optimization[propagator.lower().rstrip().lstrip()], learning_rate) if (propagator.lower().rstrip().lstrip() in optimization) else algorithm(model, optimization["adam"], learning_rate)
	elif (propagator[0].lower().rstrip().lstrip() in optimization): optimizer = algorithm(model, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, beta = propagator[1]) if (propagator[0].lower().rstrip().lstrip() == "adam") else algorithm(model, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, momentum = propagator[1]) if ((propagator[0].lower().rstrip().lstrip() == "sgd") | (propagator[0].lower().rstrip().lstrip() == "rms")) else algorithm(model, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate) if (propagator[0].lower().rstrip().lstrip() in optimization) else algorithm(model, optimization["adam"], learning_rate)
	error = cost if callable(cost) else criterion(utility[cost.lower().rstrip().lstrip()]) if (cost.lower().rstrip().lstrip() in utility) else criterion(utility["crossentropy"]) if (neurons[-1] > 1) else criterion(utility["mse"])
	if (neurons[-1] == 1): Y += list(itertools.chain.from_iterable([y for y in trainer]))
	labels = list(set(Y))
	mode = True if (TNN != []) else False
	model.train()
	os.system("cls")
	print(), print("_"*120)
	print(), print("MODEL ARCHITECTURE")
	print(), print("_"*120)
	try: print(), print(summary(model, (channel, height, height)))
	except: print(), print(model)
	print(), print("_"*120)

	for epoch in range(episodes):

		for index, pair in enumerate(trainer):

			if (TNN != []): x, y = pair[0].to(device), pair[1].to(device)
			else: x, y = pair.src.to(device), pair.trg.to(device)
			optimizer.zero_grad()
			prediction = model(x) if (TNN == []) else model(x, y[:-1])
			loss = error(prediction, y) if (TNN == []) else error(prediction.reshape(-1, prediction.shape[2]), y[1:].reshape(-1))
			loss.backward()
			optimizer.step()
			collect.append(loss.item())
			batch_error.append(loss.item())
			if (TNN != []): torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

			if (TNN == []):

				for cycle in range(len(prediction)):

					if (neurons[-1] > 1): 

						if (torch.argmax(prediction[cycle]) == y[cycle]): correct += 1 
						else: incorrect += 1

					elif (neurons[-1] == 1): 

						if (min(labels, key = lambda x: abs(x - prediction[cycle].item())) == y[cycle].item()): correct += 1
						else: incorrect += 1

				total = correct + incorrect
				ratio.append(correct / total)
				batch_accuracy.append(correct / total)
				correct, incorrect = 0, 0

		residual.append(sum(collect) / len(collect))
		if (TNN == []): accuracy.append(sum(ratio) / len(ratio))
		if (validater != []): model, deviation, score, flag = validate(model, validater, error, horizon, residual[-1], episodes, labels, mode)
		collect, ratio = [], []
		correct, incorrect = 0, 0
		if ((show == False) & (flag == True)): break

		if (show == True):

			print("\nEpisode:", epoch + 1)
			if ((validater == []) & (TNN != [])): print("Error:", round(residual[-1], 4))
			elif ((validater == []) & (TNN == [])): print("Error:", round(residual[-1], 4)), print("Accuracy:", round(accuracy[-1], 4))
			elif ((validater != []) & (TNN != [])): print("Training error:", round(residual[-1], 4)), print("Validation error:", round(deviation[-1], 4))
			else: print("Training error:", round(residual[-1], 4)), print("Validation error:", round(deviation[-1], 4)), print("Training accuracy:", round(accuracy[-1], 4)), print("Validation accuracy:", round(score[-1], 4))
			if (flag == True): break

	return model, residual, accuracy, deviation, score


def train(trainer, functions, learning_rate, episodes, cost, propagator, GAN = [], DCGAN = [], DCWGANGP = [], flatten = [], unflatten = [], show = True):

	assert (GAN != []) | (DCGAN != []) | (DCWGANGP != []), "A MODEL ARCHITECTURE IS REQUIRED"
	if (DCGAN != []): kernel, stride, padding, channels, height, pooling, convolutions, direction, offset, normalization, noise_width = DCGAN
	elif (DCWGANGP != []): kernel, stride, padding, channels, height, pooling, convolutions, direction, offset, normalization, length, lamda, noise_width = DCWGANGP
	elif (GAN != []): neurons, noise_width = GAN
	block_error, flag = [], False
	collect_generator, collect_critic, error_generator, error_critic = [], [], [], []
	decide_real, decide_fake, detect_real, detect_fake = [], [], [], []
	real_detection, fake_detection, generator_error, critic_error = [], [], [], []
	if ((type(cost) == str) & (neurons != [])): functions[0][-1], functions[1][-1] = "" if ((neurons[0][-1] > 1) & (cost.lower().rstrip().lstrip() == "crossentropy") & ("softmax" in functions[0][-1])) else functions[0][-1], "" if ((neurons[1][-1] > 1) & (cost.lower().rstrip().lstrip() == "crossentropy") & ("softmax" in functions[1][-1])) else functions[1][-1]
	generator = ConvolutionalNeuralNetwork(kernel[0], stride[0], padding[0], height, convolutions[0], functions[0], channels = channels, pooling = pooling, direction = direction[0], offset = offset[0], normalization = normalization[0], flatten = flatten[0], unflatten = unflatten[0]).to(device) if (ANN == []) else ArtificialNeuralNetwork(neurons[0], functions[0], unflatten = True, channels = channels, height = height).to(device)
	critic = ConvolutionalNeuralNetwork(kernel[1], stride[1], padding[1], height, convolutions[1], functions[1], channels = channels, pooling = pooling, direction = direction[1], offset = offset[1], normalization = normalization[1], flatten = flatten[1], unflatten = unflatten[1]).to(device) if (ANN == []) else ArtificialNeuralNetwork(neurons[1], functions[1], flatten = True, channels = channels, height = height).to(device)
	if (type(propagator) == str): optimizer_generator, optimizer_critic = algorithm(generator, optimization[propagator.lower().rstrip().lstrip()], learning_rate) if (propagator.lower().rstrip().lstrip() in optimization) else algorithm(generator, optimization["adam"], learning_rate), algorithm(critic, optimization[propagator.lower().rstrip().lstrip()], learning_rate) if (propagator.lower().rstrip().lstrip() in optimization) else algorithm(critic, optimization["adam"], learning_rate)
	elif (propagator[0].lower().rstrip().lstrip() in optimization): optimizer_generator, optimizer_critic = algorithm(generator, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, beta = propagator[1]) if (propagator[0].lower().rstrip().lstrip() == "adam") else algorithm(generator, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, momentum = propagator[1]) if ((propagator[0].lower().rstrip().lstrip() == "sgd") | (propagator[0].lower().rstrip().lstrip() == "rms")) else algorithm(generator, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate) if (propagator[0].lower().rstrip().lstrip() in optimization) else algorithm(generator, optimization["adam"], learning_rate), algorithm(critic, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, beta = propagator[1]) if (propagator[0].lower().rstrip().lstrip() == "adam") else algorithm(critic, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate, momentum = propagator[1]) if ((propagator[0].lower().rstrip().lstrip() == "sgd") | (propagator[0].lower().rstrip().lstrip() == "rms")) else algorithm(critic, optimization[propagator[0].lower().rstrip().lstrip()], learning_rate) if (propagator[0].lower().rstrip().lstrip() in optimization) else algorithm(critic, optimization["adam"], learning_rate)
	error = cost if callable(cost) else criterion(utility[cost.lower().rstrip().lstrip()]) if (cost.lower().rstrip().lstrip() in utility) else criterion(utility["bce"])
	generator.train()
	critic.train()
	os.system("cls")
	print(), print("_"*120)
	print(), print("GENERATOR MODEL ARCHITECTURE")
	print(), print("_"*120)
	try: print(), print(summary(generator, (noise_width, height, height)))
	except: print(), print(generator)
	print(), print("_"*120)
	print(), print("CRITIC MODEL ARCHITECTURE")
	print(), print("_"*120)
	try: print(), print(summary(critic, (channels, height, height)))
	except: print(), print(critic)
	print(), print("_"*120)

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
			error_critic.append(loss_critic.item()) if (length == 0) else error_critic.append(sum(block_error) / len(block_error))
			detect_real.append(decision_real.item())
			detect_fake.append(decision_fake.item())
			block_error = []

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

	return generator, critic, np.array(error_generator), np.array(error_critic), np.array(detect_real), np.array(detect_fake), np.array(generator_error), np.array(critic_error), np.array(real_detection), np.array(fake_detection)


def plot(data, colour, name, x, y, compare = False):

	plt.figure()
	axis = plt.axes(facecolor = "#E6E6E6")
	axis.set_axisbelow(True)
	plt.grid(color = "w", linestyle = "solid")
	for spine in axis.spines.values(): spine.set_visible(False)
	plt.tick_params(axis = "x", which = "both", bottom = False, top = False)
	plt.tick_params(axis = "y", which = "both", left = False, right = False)
	if (compare == True): plt.plot(list(range(1, len(data[0]) + 1)), data[0], color = f"{colour[0]}", linewidth = 1, label = f"{name[0]}"), plt.plot(list(range(1, len(data[1]) + 1)), data[1], color = f"{colour[1]}", linewidth = 1, label = f"{name[1]}")
	elif ((type(data[0]) != list) & (type(data[0]) != np.ndarray)): plt.plot(list(range(1, len(data) + 1)), data, color = f"{colour}", linewidth = 1)
	else: plt.scatter(data[0], data[1], color = f"{colour}", marker = ".", alpha = 0.2)
	plt.xlabel(f"{x}"), plt.ylabel(f"{y}")
	if (compare == True): plt.legend(loc = "best")
	plt.savefig(f"{name if (compare == False) else name[2]}.png", dpi = 200)
	plt.show()


def test(model, data, output):

	correct, incorrect, Y = 0, 0, []
	for x, y in data: Y += [y[index].item() for index in range(len(y))]
	labels = list(set(Y))
	model.eval()

	with torch.no_grad():

		for index, (x, y) in enumerate(data):

			prediction = model(x)

			for cycle in range(len(prediction)):

				if (output > 1):

					if (torch.argmax(prediction[cycle]) == y[cycle]): correct += 1 
					else: incorrect += 1

				elif (output == 1):

					if (min(labels, key = lambda x: abs(x - prediction[cycle].item())) == y[cycle].item()): correct += 1
					else: incorrect += 1

	total = correct + incorrect
	if (incorrect == 1): print(f"\nCorrectly labeled {correct} samples and incorrectly labeled {incorrect} sample")
	elif (correct == 1): print(f"\nCorrectly labeled {correct} sample and incorrectly labeled {incorrect} samples")
	else: print(f"\nCorrectly labeled {correct} samples and incorrectly labeled {incorrect} samples")
	print("\nAccuracy:", round(100*(correct / total), 2))


def evaluate(model, image_size, LUT, channel):

	camera = cv2.VideoCapture(0)
	top, right, bottom, left = 170, 150, 425, 450

	while True:

		(grabbed, frame) = camera.read()
		frame = imutils.resize(frame, width = 700)
		frame = cv2.flip(frame, 1)
		clone = frame.copy()
		(height, width) = frame.shape[:2]
		box = frame[top:bottom, right:left]
		grey = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
		grey = cv2.GaussianBlur(grey, (7, 7), 0)
		grey = cv2.resize(grey, (image_size, image_size))
		data = torch.Tensor(normalize(grey).reshape(1, channel, image_size, image_size))
		prediction = model(data)
		category = torch.argmax(graph.softmax(prediction))
		confidence = int(torch.max(graph.softmax(prediction)).item()*100)
		cv2.putText(clone, (str(LUT[category.item()])), (450, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
		cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
		cv2.putText(clone, (str(confidence) + "%"), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
		scale_percent = 70
		w, h = clone.shape[1], clone.shape[0]
		width = int(w*scale_percent / 100)
		height = int(h*scale_percent / 100)
		area = (width, height)
		resized = cv2.resize(clone, area, interpolation = cv2.INTER_AREA)
		cv2.imshow("Video Feed:", clone)
		keypress = cv2.waitKey(1) & 0xFF
		if (keypress == ord("q")): break
		elif (keypress == 27): break

	camera.release()
	cv2.destroyAllWindows()


def validate(model, validater, error, horizon, residual, episodes, labels = [], mode = False):
	
	if not all(hasattr(validate, item) for item in ["epoch", "cost_validation", "cost_training", "period", "increment", "history", "checkpoint", "cycle", "cost", "accuracy", "error_validation", "error_training"]):

		validate.epoch, validate.increment = 0, 0
		validate.cost, validate.accuracy = [], []
		validate.cost_validation, validate.cost_training = [], []
		validate.error_validation, validate.error_training = 0, 0
		validate.period = horizon + validate.increment*(horizon + 1)
		validate.history, validate.checkpoint, validate.cycle = validate.period, validate.period, validate.period

	if ((validate.epoch == 0) | (validate.epoch == validate.history + 1)): torch.save(model, "ANN.pth")
	if ((validate.epoch == 0) | (validate.epoch == validate.history + 1)): validate.history = validate.period
	flag, correct, incorrect = False, 0, 0
	batch_error, batch_accuracy = [], []
	count, counter = 0, 0
	model.eval()
	
	with torch.no_grad():

		for index, pair in enumerate(validater):

			if (mode == False): x, y = pair[0].to(device), pair[1].to(device)
			else: x, y = pair.src.to(device), pair.trg.to(device)
			prediction = model(x) if (mode == False) else model(x, y[:-1])
			loss = error(prediction, y) if (mode == False) else error(prediction.reshape(-1, prediction.shape[2]), y[1:].reshape(-1))
			batch_error.append(loss.item())
			if (mode == True): torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)

			if (mode == False):

				for iteration in range(len(prediction)):

					if (prediction.shape[-1] > 1):

						if (torch.argmax(prediction[iteration]) == y[iteration]): correct += 1
						else: incorrect += 1

					elif (prediction.shape[-1] == 1):

						if (min(labels, key = lambda x: abs(x - prediction[iteration].item())) == y[iteration].item()): correct += 1
						else: incorrect += 1

				total = correct + incorrect
				batch_accuracy.append(correct / total)
				correct, incorrect = 0, 0

	model.train()
	validate.cost.append(sum(batch_error) / len(batch_error))
	if (mode == False): validate.accuracy.append(sum(batch_accuracy) / len(batch_accuracy))
	if ((validate.epoch == 0) | (validate.epoch == validate.cycle + 1)): validate.error_validation, validate.error_training = validate.cost[-1], residual
	if ((validate.epoch == 0) | (validate.epoch == validate.cycle + 1)): validate.cycle = validate.period
	if (((validate.epoch > 0) & (validate.epoch <= horizon)) | ((validate.epoch >= validate.checkpoint + 2) & (validate.epoch <= validate.period))): validate.cost_validation.append(validate.cost[-1]), validate.cost_training.append(residual)
	if (validate.epoch == validate.period): validate.checkpoint = validate.period

	if (validate.epoch == validate.period):

		for item in range(horizon):

			if (validate.cost_training[item] < validate.error_training): counter += 1
			if (validate.cost_validation[item] >= validate.error_validation): count += 1
			elif (validate.cost_validation[item] < validate.error_validation): break

		validate.increment += 1
		validate.period = horizon + validate.increment*(horizon + 1)
		validate.cost_validation, validate.cost_training = [], []

		if ((count == horizon) & (counter >= int(0.8*horizon))): 

			print(), print("_"*120)
			print(), print("TERMINATING LEARNING PROCESS")
			print(), print(f"\nTraining summary: \n\ntruncated at {validate.epoch + 1} cycles out of {episodes}\n")
			print(), print(f"last discarded nodes;")

			for tag, parameter in model.named_parameters(): 

				try: print(f"\n\n{tag}:\n", parameter.detach().numpy(), "\n\n")	
				except Exception as e: print(e)

			model = torch.load("ANN.pth")
			print(), print(f"last reverted nodes;")

			for tag, parameter in model.named_parameters():

				try: print(f"\n\n{tag}:\n", parameter.detach().numpy(), "\n\n")	
				except Exception as e: print(e)

			print(), print("_"*120)
			flag = True

	validate.epoch += 1
	return model, validate.cost, validate.accuracy, flag


def processor(generator, critic, row, column, size, noise_width, name, flag = False):

	generator.eval(), critic.eval()
	if (flag == True): image = generator(torch.randn(size, noise_width, 1, 1).to(device))
	else: image = generator(torch.randn(size, noise_width).to(device))
	figure, axes = plt.subplots(row, column, figsize = (2*column, 2*row))
	expand = list(image.size())
	expand[0] = 1
	print()

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

def translate(model, sentence, language_input, language_output, source, maximum = 50):

	input_language_model = spacy.load(language[source.lower().lstrip().rstrip()]) if (source.lower().lstrip().rstrip() in language) else spacy.load(language["german"])
	if (type(sentence) == str): tokens = [token.text.lower() for token in input_language_model(sentence)]
	else: tokens = [token.lower() for token in sentence]
	tokens.insert(0, language_input.init_token)
	tokens.append(language_input.eos_token)
	text_to_index = [language_input.vocab.stoi[token] for token in tokens]
	guess = torch.LongTensor(text_to_index).unsqueeze(1).to(device)
	outputs = [language_output.vocab.stoi["<sos>"]]

	for index in range(maximum):

		target = torch.LongTensor(outputs).unsqueeze(1).to(device)
		with torch.no_grad(): prediction = model(guess, target)
		best = prediction.argmax(2)[-1, :].item()
		outputs.append(best)
		if (best == language_output.vocab.stoi["<eos>"]): break

	translation = [language_output.vocab.itos[item] for item in outputs]
	return translation[1:]

def bleu(model, data, language_input, language_output):

	targets, outputs = [], []

	for example in data:

		source = vars(example)["src"]
		target = vars(example)["trg"]
		prediction = translate(model, source, language_input, language_output)
		prediction = prediction[:-1]
		targets.append([target])
		outputs.append(prediction)

	return bleu_score(outputs, targets)
