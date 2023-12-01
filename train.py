import os
import data
import tqdm
import torch
import model
import config
import evaluation
import numpy as np
import torch.nn as nn
from itertools import groupby

import datetime



def train(type, name):

	print("Hola")
	
	# Obtain dictionaries
	w2i, i2w = data.new_obtain_dictionaries(type, name)
	print(w2i)
	print('---------------------------')
	print()
	
	# Create model:
	crnn = model.CRNN(height = 100, num_classes = len(w2i))

	# Create data generator:
	data_gen = data.data_generator(w2i = w2i, type = type, name = name, img_height = 100)

	# Instantiating CTC loss:
	ctc_loss = nn.CTCLoss(blank = len(w2i))

	# ADAM optimizer:
	optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

	# Fitting model:
	#with open(os.path.join(config.folds_path, 'Fold' + str(config.fold), 'Partitions', 'Train.lst')) as f:
	with open(os.path.join(config.folds_path, type, name, 'train.txt')) as f:
		total_files = len(list(map(str.strip, f.readlines())))
	
	# Run on GPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	crnn.to(device)
	print(f"Device {device}")


	# Train loop:
	loss_acc = []
	for it_epoch in range(config.n_epochs):
		crnn.train()
		for _ in tqdm.tqdm(range(total_files//config.batch_size), leave=True):
			
			# Resetting optimizer:
			optimizer.zero_grad()
			
			# Loading data:
			img, seq, img_length, seq_length = next(data_gen)

			#print(seq)

			# Passing through the CRNN:
			input = crnn(img.float()) # .double()
			
			# Computing loss:
			input = input.permute(1, 0, 2)
			loss = ctc_loss(input, seq, img_length, seq_length)

			# Propagating loss:
			loss.backward()

			# Optimization step:
			optimizer.step()

		loss_acc.append(loss.cpu().detach().item())
		print(f"Epoch {it_epoch + 1} ; Total loss: {loss_acc[-1]}")

		# Evaluating model:
		crnn.eval()
		###-Validation:
		SER_val = evalute_partition(w2i = w2i, pred_model = crnn, type = type, name = name, img_height = 100, partition = 'valid')
		#SER_val = evalute_partition(w2i = w2i, pred_model = crnn, img_height = 100, partition = 'ValidationLines')

		###-Test:
		SER_test = evalute_partition(w2i = w2i, pred_model = crnn, type = type, name = name, img_height = 100, partition = 'test')
		#SER_test = evalute_partition(w2i = w2i, pred_model = crnn, img_height = 100, partition = 'TestLines')
		print("Epoch #{}\tVAL: {:.2f}%\tTEST: {:.2f}%".format(it_epoch+1, 100*SER_val, 100*SER_test))

	if not os.path.exists('Modelos'):
   		# Create a new directory because it does not exist
		os.makedirs('Modelos')

	if not os.path.exists(os.path.join('Modelos', name)):
   		# Create a new directory because it does not exist
		os.makedirs(os.path.join('Modelos', name))
		   
	PATH = os.path.join('Modelos', name, datetime.date.today().strftime("%m-%d-%Y_%H-%M-%S") + '.pt') 
	torch.save(crnn, PATH)

	return




def decode_CTC(batch_posteriorgram, input_length):
	out = list()

	for it_batch in range(len(batch_posteriorgram)):
		#Performing Best Path decoding:
		decoded_sequence = list()
		# posteriorgram = batch_posteriorgram[it_batch][0:original_batch_images_size[it_batch]]
		posteriorgram = batch_posteriorgram[it_batch]

		total_timesteps = len(posteriorgram[:input_length[it_batch],:])
		total_estimation = 0
		
		#Looping over temporal slices to analyze:
		for array in posteriorgram[:input_length[it_batch],:]:
			#Estimated symbol:
			#print(array)
			decoded_value = [np.where(array.max() == array)[0][0] if np.where(array.max() == array)[0][0] != len(array) -1 else -1]
			#print(array.max())
			total_estimation += 1 + array.max()

			#Appending symbol:
			decoded_sequence.extend(decoded_value)
		
		mean_estimation = total_estimation / total_timesteps
		#print(mean_estimation)
		#Applying function B for grouping alike symbols:
		decoded_sequence = [i[0] for i in groupby(decoded_sequence) if i[0] != -1]

		#print(decoded_sequence)

		out.append(decoded_sequence)
	return out



def evalute_partition(w2i, pred_model, type, name, img_height = 50, partition = 'Test'):
	#with open(os.path.join(config.folds_path, 'Fold' + str(config.fold), 'Partitions', partition + '.lst')) as f:
	with open(os.path.join(config.folds_path, type, name, partition + '.txt')) as f:
		files = list(map(str.strip, f.readlines()))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

	it = 0
	y_true = list()
	y_pred = list()
	while it < len(files):
		init_idx = it
		end_idx = min(it + config.batch_size, len(files))
		img_batch, gt_batch, input_length, _ = data.load_batch_data(w2i, files[init_idx:end_idx], img_height, type, name)


		batch_posteriorgram = pred_model(torch.from_numpy(np.array(img_batch)).float().to(device))
		prediction = decode_CTC(batch_posteriorgram.cpu().detach().numpy(), input_length)

		#print("Prediction: ", prediction)
		#print("True      : ", gt_batch)

		y_pred.extend(prediction)
		y_true.extend(gt_batch)

		it = end_idx

	ser = evaluation.SER(y_true = y_true, y_pred = y_pred)

	return ser


if __name__ == '__main__':
	train()
