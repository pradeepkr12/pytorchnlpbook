from utils.utils import handle_dirs, set_seed_everywhere
from argparse import Namespace
import torch

from data_preprocessing.data_prep import prepare_data
# --- config details

# Data and Path information
args = Namespace(
    frequency_cutoff=25,
    model_state_file='model.pth',
    review_csv='/Users/pradeepkumarmahato/pradeep/nlp/github/PyTorchNLPBook/data/yelp/reviews_with_splits_lite.csv',
    # review_csv='data/yelp/reviews_with_splits_full.csv',
    save_dir='data/model_storage/ch3/yelp/',
    vectorizer_file='vectorizer.json',
    # No Model hyper parameters
    # Training hyper parameters
    batch_size=128,
    early_stopping_criteria=5,
    learning_rate=0.001,
    num_epochs=100,
    seed=1337,
    # Runtime options
    catch_keyboard_interrupt=True,
    cuda=True,
    expand_filepaths_to_save_dir=True,
    reload_from_files=False,
)

# Check CUDA
if not torch.cuda.is_available():
    args.cuda = False

print("Using CUDA: {}".format(args.cuda))

args.device = torch.device("cuda" if args.cuda else "cpu")

# Set seed for reproducibility
set_seed_everywhere(args.seed, args.cuda)

# handle dirs
handle_dirs(args.save_dir)
# -- config ends ----------

# -- data prep --
dataset, vectorizer =  prepare_data(args)
