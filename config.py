import os
import sys

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
n_epochs = 100

img_height = 128
confidence_threshold_max = 0.9
confidence_threshold_min = 0.2

confidence_limit = 5

patience = 15