import argparse
import os
import sys

root_dir = os.path.join(os.getcwd(), '..')
sys.path.append(root_dir)

# Import Project Modules
from src.models.gan import GAN
from src.models.fbgan import FB_GAN
from utils.data_utilities import prepare_dataset
from globals import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--desired-features",
        type=str,
        nargs="+",
        default=DESIRED_FEATURES,
        choices=['B', 'C', 'E', 'G', 'H', 'I', 'S','T'],
    )
    parser.add_argument(
        "--task-type",
        type=str,
        nargs="+",
        default="DNA",
        choices=["DNA", "protein"],
    )
    parser.add_argument(
        "--max-len",
        type=int,
        nargs="+",
        default=243,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=PATH_DATA,
    )
    args = parser.parse_args()

    inputs = prepare_dataset(args.data_path)

    # Pre-train GAN
    gan = GAN()
    gan.train(inputs = inputs,
              epochs = 5,
              step_log = 10)

    # Load Feedback-GAN model
    ganfb = FB_GAN(generator_path=PATH_G,
                   discriminator_path=PATH_D,
                   fbnet_path=PATH_FB,
                   features=args.desired_features,
                   multip_factor=50,
                   log_id='trial')

    # Saves logs in a separate directory
    ganfb.train(inputs,
                step_log=1,
                epochs=1,
                steps_per_epoch=3)





