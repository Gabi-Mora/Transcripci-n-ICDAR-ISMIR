import data
import config
import torch

import argparse
import os
import numpy as np
import datetime
import csv

import train
import evaluation

if __name__ == '__main__':

	desc = "Transcription of text/musical documents using CRNN-CTC model"

	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument("-type", help = "Type of dataset: musica/texto")
	parser.add_argument("-model", help = "Model use for the test")
	parser.add_argument("-input", help = "Images to use as test samples")
	args = parser.parse_args()

	type = model = input = None

	if args.type and args.model and args.input:
		print("Type: % s" % args.type)
		type = args.type
		print("Model: % s" % args.model)
		model = args.model
		print("Test Input: % s" % args.input)
		input = args.input
	else:
		raise Exception("Exception")
	
	w2i, i2w = data.new_obtain_dictionaries(type, input)
	print(w2i)

	crnn = torch.load(args.model)

	with open(os.path.join(config.folds_path, type, input, 'test.txt')) as f:
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
		img_batch, gt_batch, input_length, _ = data.load_batch_data(w2i, files[init_idx:end_idx], config.img_height, type, input)

		batch_posteriorgram = crnn(torch.from_numpy(np.array(img_batch)).float().to(device))
		prediction, confidence_list = train.decode_CTC(batch_posteriorgram.cpu().detach().numpy(), input_length)

		files_names += files[init_idx:end_idx]
		confidences += confidence_list

		y_pred.extend(prediction)
		y_true.extend(gt_batch)

		it = end_idx

	ser = evaluation.SER(y_true = y_true, y_pred = y_pred)
	print("TEST: {:.2f}%".format(100*ser))
	
	if not os.path.exists('Confianzas'):
   		# Create a new directory because it does not exist
		os.makedirs('Confianzas')

	if not os.path.exists(os.path.join('Confianzas', type)):
   		# Create a new directory because it does not exist
		os.makedirs(os.path.join('Confianzas', type))

	nombre_modelo = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
	csv_file = os.path.join('Confianzas', type, nombre_modelo + '.csv')

	with open(csv_file, 'w', newline='') as file:
		writer = csv.writer(file)
		for i in range(len(files_names)):
			writer.writerow([files_names[i], confidences[i]])

	print('Confidences saved at: ', csv_file)

	