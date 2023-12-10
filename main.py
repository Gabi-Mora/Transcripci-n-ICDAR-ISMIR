import data
import config
import torch
import model

import argparse

import train




if __name__ == '__main__':

	desc = "Transcription of text/musical documents using CRNN-CTC model"

	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument("-type", help = "Type of dataset: musica/texto")
	parser.add_argument("-name", help = "Name of dataset")
	parser.add_argument("-rate", help = "Initial rate of images used in training")
	args = parser.parse_args()

	type = name = rate = None

	if args.type and args.name:
		print("Type: % s" % args.type)
		type = args.type
		print("Name: % s" % args.name)
		name = args.name
	else:
		raise Exception("Exception")
	
	if args.rate:
		rate = args.rate
	
	train.train(type, name, rate)

	print("Hello")

	