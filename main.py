import data
import config
import torch
import model

import argparse

import train




if __name__ == '__main__':

	desc = "Transcription of text/musical documents using CRNN-CTC model"

	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument("-type",	help = "Type of dataset: musica/texto")
	parser.add_argument("-name1",	help = "Main dataset")
	parser.add_argument("-name2",	help = "Secondary dataset used on the Out of Domain Training")
	parser.add_argument("-rate",	help = "Initial rate of images to use in training")
	parser.add_argument("-process",	help = "Type of algorithm to use in training (1: Out of domain, 2: On Domain 3: ISMIR)")
	args = parser.parse_args()

	type = name1 = name2 = rate = process = None

	if args.type and args.name1 and args.process:
		print("Type: % s" % args.type)
		type = args.type
		print("Name: % s" % args.name1)
		name1 = args.name1
		print("Process: % s" % args.process)
		process = int(args.process)
	else:
		raise Exception("Exception1")
	
	if args.rate:
		rate = args.rate

	if process == 1:
		if args.name2:
			name2 = args.name2
		else:
			raise Exception("Exception2")

	
	train.train(type, name1, name2, rate, process)

	print("Hello")

	