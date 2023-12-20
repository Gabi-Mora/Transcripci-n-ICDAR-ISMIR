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

import csv



def train(type, name, rate):

	print("Hola")
	
	# Obtain dictionaries
	w2i, i2w = data.new_obtain_dictionaries(type, name)
	print(w2i)
	print('---------------------------')
	print()

	if rate is not None:
		data.create_fold(type = type, name = name, partition = "train", rate = rate)
		data.create_fold(type = type, name = name, partition = "valid", rate = rate)
	
	# Create model:
	crnn = model.CRNN(height = config.img_height, num_classes = len(w2i))

	# Create data generator:
	data_gen = data.data_generator(w2i = w2i, type = type, name = name, rate = rate, img_height = config.img_height)

	# Instantiating CTC loss:
	ctc_loss = nn.CTCLoss(blank = len(w2i), zero_infinity=True)

	# ADAM optimizer:
	optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

	best_SER = float.MAX_VALUE
	count = 0

	nombre_modelo = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

	# Fitting model:
	#with open(os.path.join(config.folds_path, 'Fold' + str(config.fold), 'Partitions', 'Train.lst')) as f:
	#with open(os.path.join(config.folds_path, type, name, 'train.txt')) as f:
	#	total_files = len(list(map(str.strip, f.readlines())))
	#	print(total_files)

	with open(os.path.join(config.folds_path, type, name, rate, 'train.txt')) as f:
		total_files = len(list(map(str.strip, f.readlines())))
		print(total_files)
	
	# Run on GPU
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
	crnn.to(device)
	print(f"Device {device}")

	if not os.path.exists('Modelos'):
   		# Create a new directory because it does not exist
		os.makedirs('Modelos')

	if not os.path.exists(os.path.join('Modelos', name)):
   		# Create a new directory because it does not exist
		os.makedirs(os.path.join('Modelos', name))

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
		SER_val = evalute_partition(w2i = w2i, pred_model = crnn, type = type, name = name, rate = rate, img_height = config.img_height, partition = 'valid')

		###-Test:
		#SER_test = evalute_partition(w2i = w2i, pred_model = crnn, type = type, name = name, img_height = config.img_height, partition = 'test')
		#SER_test = evalute_partition(w2i = w2i, pred_model = crnn, img_height = config.img_height, partition = 'TestLines')
		#print("Epoch #{}\tVAL: {:.2f}%\tTEST: {:.2f}%".format(it_epoch+1, 100*SER_val, 100*SER_test))

		print("Epoch #{}\tVAL: {:.2f}%".format(it_epoch+1, 100*SER_val))
		   
		# Checkpoint
		if SER_val < best_SER:
			PATH = os.path.join('Modelos', name, nombre_modelo + '.pt')
			torch.save(crnn, PATH)
			count = 0
		else:
			count += 1

		# EarlyStopping
		if count == config.patience:
			print('Training Stopped: EarlyStopping')
			break

	#SER_test = evalute_partition(w2i = w2i, pred_model = crnn, type = type, name = name, img_height = config.img_height, partition = 'test')

	#print("TEST: {:.2f}%".format(100*SER_test))

	return




def decode_CTC(batch_posteriorgram, input_length):
	out = list()
	confidences = list()

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
			decoded_value = [np.where(array.max() == array)[0][0] if np.where(array.max() == array)[0][0] != len(array) -1 else -1]
			total_estimation += 1 + array.max()

			#Appending symbol:
			decoded_sequence.extend(decoded_value)
		
		mean_estimation = total_estimation / total_timesteps
		
		#Applying function B for grouping alike symbols:
		decoded_sequence = [i[0] for i in groupby(decoded_sequence) if i[0] != -1]

		#print(decoded_sequence)

		out.append(decoded_sequence)
		confidences.append(mean_estimation)

	return out, confidences



def evalute_partition(w2i, pred_model, type, name, rate, img_height = 50, partition = 'Test'):
	new_name = None
	if rate is not None:
		new_name = name + '/' + rate
	else:
		new_name = name

	with open(os.path.join(config.folds_path, type, new_name, partition + '.txt')) as f:
		files = list(map(str.strip, f.readlines()))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

	it = 0
	y_true = list()
	y_pred = list()

	files_names = list()
	confidences = list()
	
	while it < len(files):
		init_idx = it
		end_idx = min(it + config.batch_size, len(files))
		img_batch, gt_batch, input_length, _ = data.load_batch_data(w2i, files[init_idx:end_idx], img_height, type, name)

		batch_posteriorgram = pred_model(torch.from_numpy(np.array(img_batch)).float().to(device))
		prediction, confidence_list = decode_CTC(batch_posteriorgram.cpu().detach().numpy(), input_length)

		files_names += files[init_idx:end_idx]
		confidences += confidence_list

		y_pred.extend(prediction)
		y_true.extend(gt_batch)

		it = end_idx

	ser = evaluation.SER(y_true = y_true, y_pred = y_pred)

	#if rate is not None and partition == 'test':
	if partition == 'test':
		if not os.path.exists('Confianzas'):
   			# Create a new directory because it does not exist
			os.makedirs('Confianzas')

		if not os.path.exists(os.path.join('Confianzas', name)):
   			# Create a new directory because it does not exist
			os.makedirs(os.path.join('Confianzas', name))

		nombre_modelo = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
		csv_file = os.path.join('Confianzas', name, nombre_modelo + '.csv')

		with open(csv_file, 'w', newline='') as file:
			writer = csv.writer(file)
			for i in range(len(files_names)):
				writer.writerow([files_names[i], confidences[i]])

		print('Confidences saved at: ', csv_file)

	return ser

def confidence_rate(batch_posteriorgram, input_length):
	for it_batch in range(len(batch_posteriorgram)):
		#Performing Best Path decoding:
		posteriorgram = batch_posteriorgram[it_batch]

		total_timesteps = len(posteriorgram[:input_length[it_batch],:])
		total_estimation = 0
		
		#Looping over temporal slices to analyze:
		for array in posteriorgram[:input_length[it_batch],:]:
			total_estimation += 1 + array.max()
		
		mean_estimation = total_estimation / total_timesteps

	return mean_estimation


if __name__ == '__main__':
	train()
