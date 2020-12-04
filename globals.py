import os

####################################################################################
# GLOBAL VARIABLES
####################################################################################

DESIRED_FEATURES = ['C', 'H', 'E']

MAX_LEN = 243
MIN_LEN = 0

TASK_TYPE = 'DNA'
MAX_LEN_PROTEIN = MAX_LEN // 3 - 6
MIN_LEN_PROTEIN = MIN_LEN // 3 - 6

SEQ_LENGTH = MAX_LEN
DIM = 50
KERNEL_SIZE = 5
BATCH_SIZE = 128
N_CHAR = 5
NOISE_SHAPE = 128

MAX_LEN_FB = 128  # max length of sequence we want to consider for training
n_tags = 8  # number of classes in 8-state prediction
n_words = 8

# UPDATE THE PATHS

# Select a path where your data are stores
PATH_DATA = '2018-06-06-ss.cleaned.csv'

# Select paths to the saved weights of the gan and feedback
PATH_G = 'Weights/Generator'
PATH_D = 'Weights/Discriminator'
PATH_FB = 'Weights/Feedback'