import os
import random
import numpy as np
from shutil import copyfile, rmtree
from captcha.image import ImageCaptcha

def gen_rand():

	#POSSIBLE VOCABULARY:
	#vocab='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
	#vocab='0123456789abcdefghijklmnopqrstuvwxyz'
	vocab='0123456789'
	#vocab='0123456789abcdefg'

	buf = ""
	max_len = random.randint(3,7)
	for i in range(max_len):
		buf += random.choice(vocab)
	return buf


def generateImg(ind, imgPath, gtPath):
	captcha = ImageCaptcha(fonts=['./fonts/roboto.ttf'])
	theChars = gen_rand()
	captcha._height = random.randint(50,100)
	captcha._width = random.randint(100,200)
	data = captcha.generate(theChars)
	imgName = str(ind) + '.png'
	gtName = str(ind) + '.txt'
	
	#Writing image:
	captcha.write(theChars, os.path.join(imgPath, imgName))

	#Writing GT:
	with open(os.path.join(gtPath, gtName), 'w') as f:
		f.write("\t".join([u for u in theChars]))
	

def generateConfigurationYAML(root_images_path):
	with open(os.path.join(root_images_path, 'Captcha_images.yml'), 'w') as f:
		f.write('name : Catpcha_images\n')
		f.write('path_to_corpus : ./Captcha_images/\n')
		f.write('image_extension : .png\n')
		f.write('GT_extension : .txt\n')
		f.write('user_max_height : 50\n')
		f.write('user_max_width : -1\n')
		f.write('batch_size : 16\n')
		f.write('epochs : 1\n')
		f.write('super_epochs : 2000\n')
		f.write('create_dictionaries : True\n')
		f.write('\n')
		f.write('LM:\n')
		f.write('  create : True\n')
		f.write("  n_value : 2\n")
		f.write('\n')
		f.write("CTC_decoding:\n")
		f.write("  type : beam_search\n")
		f.write("  param: 3\n")
		f.write('\n')
		f.write('architecture:\n')
		f.write('  #Convolutional stages:\n')
		f.write('  filters : [32, 64]\n')
		f.write('  kernel_size : [[3,3], [3,3]]\n')
		f.write('  pool_size : [[2,2], [2,2]]\n')
		f.write('  pool_strides : [[2, 2], [2, 2]]\n')
		f.write('  activations : [\'LeakyReLU\', \'LeakyReLU\']\n')
		f.write('  param_activation : [0.2, 0.2]\n')
		f.write('  batch_norm : [True, True]\n')
		f.write('  #Recurrent stages:\n')
		f.write('  units : [128, 128]\n')
		f.write('  dropout : [0, 0]\n')
		f.write('\n')
		f.write('\n')
		f.write('optimizer : K.optimizers.Adam()\n')
		f.write('\n')




	return


def run(num_images, root_images_path):


	# Generate images and GT in the data directory:
	###-Creating Data directory:
	data_directory = os.path.join(root_images_path, 'Data')
	if not os.path.exists(data_directory):
		os.mkdir(data_directory)
	###-Creating Image directory
	image_directory = os.path.join(data_directory, 'Images')
	if not os.path.exists(image_directory):
		os.mkdir(image_directory)

	###-Creating GT directory
	###-Creating Image directory
	gt_directory = os.path.join(data_directory, 'GT')
	if not os.path.exists(gt_directory):
		os.mkdir(gt_directory)

	###-Generating images:
	for u in range(num_images):
		print("{} out of {}".format(u+1,num_images))
		generateImg(u, image_directory, gt_directory)

	return

def create_cv_dict(num, cv, root_path):

	root_folds_path = os.path.join(root_path,'Folds')

	init_list = list(range(num))
	random.shuffle(init_list)

	cv_list = np.split(np.array(init_list), cv)
	for single_fold in range(cv):
		partitions_path = os.path.join(root_folds_path, 'Fold' + str(single_fold), 'Partitions')

		test_partition = single_fold
		validation_partition = (single_fold + 1)%cv
		train_partitions = list(set(range(cv)) - {test_partition} - {validation_partition})
		# print("Creating Fold {}".format(single_fold+1))
		# print("\tTrain partition:\t{}".format(train_partitions))
		# print("\tValidation partition:\t{}".format(validation_partition))
		# print("\tTest partition:\t\t{}".format(test_partition))

		#Writing partition files:
		###-Training partition:
		train_elements = [cv_list[u] for u in train_partitions]
		train_elements = [item for sublist in train_elements for item in sublist]
		with open(os.path.join(partitions_path, 'Train.lst'),'w') as f:
			for u in train_elements:
				f.write(str(u) + "\n")


		###-Test partition:
		test_elements = list(cv_list[test_partition])
		with open(os.path.join(partitions_path, 'Test.lst'),'w') as f:
			for u in test_elements:
				f.write(str(u) + "\n")

		###-Val partition:
		val_elements = list(cv_list[validation_partition])
		with open(os.path.join(partitions_path, 'Val.lst'),'w') as f:
			for u in val_elements:
				f.write(str(u) + "\n")

	return


def create_folder_structure(cv, root_images_path):
	partitions = ['train','val','test']

	if not os.path.exists(root_images_path):
		os.mkdir(root_images_path)
	root_folds_path = os.path.join(root_images_path,'Folds')
	if not os.path.exists(root_folds_path):
		os.mkdir(root_folds_path)	

	for it in range(cv):
		cv_fold = os.path.join(root_folds_path, 'Fold' + str(it))
		if not os.path.exists(cv_fold):
			os.mkdir(cv_fold)		 
		partitions_path = os.path.join(cv_fold, 'Partitions')
		if not os.path.exists(partitions_path):
			os.mkdir(partitions_path)

if __name__=='__main__':
	root_images_path = 'Captcha_images'
	num_images = 2000
	cv_folds = 5

	# Generating folder structure:
	create_folder_structure(cv = 5, root_images_path = root_images_path)
	
	# Generate configuration file:
	generateConfigurationYAML(root_images_path = root_images_path)

	# Create data dictionary:
	create_cv_dict(num = num_images, cv = cv_folds, root_path = root_images_path)

	# Run image generation:
	run(num_images, root_images_path)