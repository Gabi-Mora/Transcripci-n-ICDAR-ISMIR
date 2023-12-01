import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
	def __init__(self, height, num_classes):
		super(CRNN, self).__init__()

		self.height = height
		self.n_classes = num_classes

		CNN_layers = [
			nn.Conv2d(in_channels = 1, 	out_channels = 32, 	kernel_size = (3, 3), padding = 'same'),
			nn.BatchNorm2d(num_features = 32),
			nn.LeakyReLU(negative_slope = 0.2),
			nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 1)),

			nn.Conv2d(in_channels = 32, out_channels = 64, 	kernel_size = (3, 3), padding = 'same'),
			nn.BatchNorm2d(num_features = 64),
			nn.LeakyReLU(negative_slope = 0.2),
			nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 1)),
			
			nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), padding = 'same'),
			nn.BatchNorm2d(num_features = 128),
			nn.LeakyReLU(negative_slope = 0.2),
			nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),

			nn.Conv2d(in_channels = 128, out_channels = 256,kernel_size = (3, 3), padding = 'same'),
			nn.BatchNorm2d(num_features = 256),
			nn.LeakyReLU(negative_slope = 0.2),
			nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2)),

		]
		self.CNN = nn.Sequential(*CNN_layers)

		RNN_layers = [
			nn.LSTM(input_size = 256*self.height//4, hidden_size = 128, bidirectional = True, batch_first = True),
		]
		self.RNN = nn.Sequential(*RNN_layers)

		dense_layers = [
			nn.Linear(in_features = 128, out_features = self.n_classes + 1), # Additional one for blank symbol
			nn.LogSoftmax(dim = 2)
		]
		self.dense = nn.Sequential(*dense_layers)
		

	def forward(self, x):
		x = self.CNN(x)
		x = x.permute(0, 3, 2, 1)
		x = x.reshape(x.shape[0], -1, x.shape[2] * x.shape[3])
		x, _ = self.RNN(x)
		return self.dense(x)
	
	def load_model(self, path):
		model = torch.load(path)
		return model

if __name__ == '__main__':
	# batch, channels, height, width
	tensor = torch.tensor(np.ones(shape = (16, 1, 100, 150)))
	model = CRNN(height = 100, num_classes = 100)
	print("hello")