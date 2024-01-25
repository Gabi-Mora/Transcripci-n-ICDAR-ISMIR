import os, shutil
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
import sys

import evaluate_test
import tempfile
import time

import glob



def train(type, name, name2, rate, term):

	if not os.path.exists('Temp'):
   		# Create a new directory because it does not exist
		os.makedirs('Temp')
		os.makedirs('Temp/Folds')
		os.makedirs('Temp/GT')
	else:
		shutil.rmtree('Temp')
		os.makedirs('Temp')
		os.makedirs('Temp/Folds')
		os.makedirs('Temp/GT')
	
	# Obtain dictionaries
	w2i, i2w = data.new_obtain_dictionaries(type, name)
	print(w2i)
	print('---------------------------')
	print()

	if term == 1:
		data.create_fold(type = type, name = name, partition = "train", rate = 100)
		data.create_fold(type = type, name = name, partition = "valid", rate = 100)
		data.create_fold(type = type, name = name2, partition = "train", rate = 0)
		data.create_fold(type = type, name = name2, partition = "valid", rate = 0)

		data.create_fold(type = type, name = name, partition = "test", rate = 100)
		data.create_fold(type = type, name = name2, partition = "test", rate = 100)

		define_temp_gt(type = type, name = name)
		define_temp_gt(type = type, name = name)
		define_temp_gt(type = type, name = name2)
		define_temp_gt(type = type, name = name2)
	elif term <= 3:
		data.create_fold(type = type, name = name, partition = "train", rate = rate)
		data.create_fold(type = type, name = name, partition = "valid", rate = rate)

		data.create_fold(type = type, name = name, partition = "test", rate = 100)

		define_temp_gt(type = type, name = name)
		define_temp_gt(type = type, name = name)
	else:
		raise("Exception")

	# Create model:
	crnn = model.CRNN(height = config.img_height, num_classes = len(w2i))

	# Create data generator:
	data_gen = data.data_generator(w2i = w2i, type = type, name = name, rate = rate, img_height = config.img_height)

	# Instantiating CTC loss:
	ctc_loss = nn.CTCLoss(blank = len(w2i), zero_infinity=True)

	# ADAM optimizer:
	optimizer = torch.optim.Adam(crnn.parameters(), lr=0.001)

	best_SER = sys.float_info.max
	count = 0

	nombre_modelo = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
	loop = 1
	
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

	name_rate = None
	if rate is None:
		name_rate = name
	else:
		name_rate = name + '/' + 'rate'

	end_loop = False

	while end_loop is not True:

		#with open(os.path.join(config.folds_path, type, name_rate, 'train.txt')) as f:
		with open(os.path.join('Temp', 'Folds', 'train.txt')) as f:
			total_files = len(list(map(str.strip, f.readlines())))
			print("Total files: ", total_files)

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
		
		if rate is not None:
			end_loop = create_diccionary(crnn, w2i, i2w, type, name, term, 'train.txt')
			_		 = create_diccionary(crnn, w2i, i2w, type, name, term, 'valid.txt')

			if not os.path.exists('Confianzas'):
   				# Create a new directory because it does not exist
				os.makedirs('Confianzas')

			if not os.path.exists(os.path.join('Confianzas', name)):
   				# Create a new directory because it does not exist
				os.makedirs(os.path.join('Confianzas', name))

			if not os.path.exists(os.path.join('Confianzas', name, nombre_modelo)):
   				# Create a new directory because it does not exist
				os.makedirs(os.path.join('Confianzas', name, nombre_modelo))

			csv_file = os.path.join('Confianzas', name, nombre_modelo, str(loop) + '.csv')

			with open(os.path.join('Temp', 'Folds', 'test.txt')) as f:
				test_files = list(map(str.strip, f.readlines()))

			files_names, confidences, y_pred, y_true = evaluate_test.evaluation_loop(crnn, w2i, test_files, type, name)

			dicc = {}
			for i in range(len(files_names)):
				trans_pred = list()
				trans_true = list()

				Value_SER = evaluation.SER([y_true[i]], [y_pred[i]])

				for token in y_true[i]:
					if token != -1:
						trans_true.append(i2w[token])

				for token in y_pred[i]:
					trans_pred.append(i2w[token])

				#dicc[files_names[i]] = [confidences[i], y_pred[i]]
				dicc[files_names[i]] = [confidences[i], trans_pred, trans_true, Value_SER]

			if term != 3:
				dicc = {k: v for k, v in sorted(dicc.items(), key=lambda item: item[1], reverse=True)}

			with open(csv_file, 'w', newline='') as file:
				writer = csv.writer(file)
				#for i in range(len(files_names)):
				index = list(dicc)
				writer.writerow(['Nombre Imagen', 'Confianza', 'Secuencia Predecida', 'Secuencia Verdadera', 'SER'])
				for x in index:
					value = dicc[x]
					writer.writerow([x, value[0], value[1], value[2], value[3]])

			print('Confidences saved at: ', csv_file)
			loop += 1
			
	shutil.rmtree('Temp')

	return

def define_temp_gt(type, name):
	ext = str()
	if type == "musica":
		ext = '.png.agnostic'
	else:
		if name == "Bentham":
			ext = '.txt'
		else:
			ext = '.png.txt'

	files = glob.glob(os.path.join(config.DDBB_path, type, name, '*' + ext))

	for x in files:
		with open(x) as f:
			line = f.readlines()[0]
			ff = open(os.path.join('Temp', 'GT', x.split('/')[-1].split('.')[0] + '.txt'), "w")
			ff.write(line)
			ff.close()

def create_diccionary(crnn, w2i, i2w, type, name, term, file_name):
	eval_files = list()
	#with open(os.path.join(config.folds_path, type, name, 'rate', 'res_train.txt')) as f:
	with open(os.path.join('Temp', 'Folds', 'res_' + file_name)) as f:
		eval_files += list(map(str.strip, f.readlines()))

	files_names, confidences, y_pred, y_true = evaluate_test.evaluation_loop(crnn, w2i, eval_files, type, name)

	dicc = {}
	for i in range(len(files_names)):
		dicc[files_names[i]] = [confidences[i], y_pred[i]]
		
	if term != 3:
		dicc = {k: v for k, v in sorted(dicc.items(), key=lambda item: item[1], reverse=True)}
		#print(dicc)
	else:
		dicc = {k: v for k, v in sorted(dicc.items(), key=lambda item: item[1], reverse=False)}


	index = list(dicc)
	add_files = list()

	#f = open(os.path.join(config.folds_path, type, name, 'rate', 'train.txt'), "a")
	f = open(os.path.join('Temp', 'Folds', file_name), "a")
	for x in index:
		value = dicc[x]

		if term != 3:
			if value[0] > config.confidence_threshold_max:
				add_files.append(x)
				f.write(x + '\n')

				if term != 3:
					ff = open(os.path.join('Temp', 'GT', x.split('/')[-1] + '.txt'), "w")
					for token in value[1]:
						ff.write(str(i2w[token]) + '\t')
					ff.write('\n')
					ff.close()
				else:
					break
		else:
			if value[0] < config.confidence_threshold_min:
				add_files.append(x)
				f.write(x + '\n')

				if term != 3:
					ff = open(os.path.join('Temp', 'GT', x.split('/')[-1] + '.txt'), "w")
					for token in value[1]:
						ff.write(str(i2w[token]) + '\t')
					ff.write('\n')
					ff.close()
				else:
					break
				
	f.close()

	end_loop = True
	f = open(os.path.join('Temp', 'Folds', 'res_' + file_name), "w")
	for x in eval_files:
		delete = False
		for y in add_files:
			if x == y:
				delete = True
				end_loop = False
				
		if not delete:
			f.write(x + '\n')

	f.close()

	return end_loop


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

		out.append(decoded_sequence)
		confidences.append(mean_estimation)

	return out, confidences



def evalute_partition(w2i, pred_model, type, name, rate, img_height = 50, partition = 'Test'):
	
	with open(os.path.join('Temp', 'Folds', partition + '.txt')) as f:
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
