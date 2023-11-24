import data
import config
import torch
from model import CRNN

import argparse

import train




if __name__ == '__main__':

	"""
	w2i, i2w = data.obtain_dictionaries()
	data_gen = data.data_generator(w2i = w2i, img_height = 100)

	img, seq, img_len, seq_len = next(data_gen)
	img = torch.tensor(img)
	model = CRNN(height = 100, num_classes = len(w2i))
	"""

	msg = "Description"

	parser = argparse.ArgumentParser(description = msg)
	parser.add_argument("-type", help = "Type of dataset: musica/texto")
	parser.add_argument("-name", help = "Name of dataset")
	args = parser.parse_args()

	type = name = None

	if args.type and args.name:
		print("Type: % s" % args.type)
		type = args.type
		print("Name: % s" % args.name)
		name = args.name
	else:
		raise Exception("Exception")
	
	train.train(type, name)

	print("Hello")