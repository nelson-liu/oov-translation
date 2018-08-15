from __future__ import division
import argparse
import logging
import os
import sys

import dill
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Given a checkpoint, remove the input key "
                     "from the solver init params."))
    argparser.add_argument("--checkpoint_path", type=str, required=True,
                           help=("The path to checkpoint to edit the "
                                 "solver init params."))
    argparser.add_argument("--key_to_remove", type=str, required=True,
                           help=("The key to remove from the solver init params."))

    config = argparser.parse_args()
    logger.info("Editing checkpoint at {}".format(config.checkpoint_path))
    loaded_checkpoint = torch.load(config.checkpoint_path)
    loaded_checkpoint["solver_init_params"].pop(config.key_to_remove)
    logger.info("Saving checkpoint to {}".format(config.checkpoint_path))
    torch.save(loaded_checkpoint, config.checkpoint_path, pickle_module=dill)


if __name__ == "__main__":
    # Redirect things to stdout, since we want to keep stderr clean for
    # hyperparameter optimzation
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
