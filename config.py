import os
import sys

"""
base_path = '.'
DDBB_path = os.path.join(base_path, 'Captcha_images')
folds_path = os.path.join(DDBB_path, 'Folds')
data_path = os.path.join(DDBB_path, 'Data')
"""

"""
base_path = '/Users/rafael/Desktop/TFM/Paper/datasets/texto'
DDBB_path = os.path.join(base_path, 'Bentham')
folds_path = '/Users/rafael/Desktop/TFM/Paper/Folds/texto/Bentham'
data_path = DDBB_path
"""

base_path   = '.'
DDBB_path   = os.path.join(base_path, 'Datasets')
folds_path  = os.path.join(DDBB_path, 'Folds')
data_path   = os.path.join(DDBB_path, 'Data')
musica_path = os.path.join(DDBB_path, 'musica')
texto_path  = os.path.join(DDBB_path, 'texto')

fold        = 0
data_type   = None
data_name   = None

batch_size = 16
n_epochs = 150