import argparse
from copy import deepcopy
import json
import logging
import os
from six.moves import input
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from oov.models import implemented_models

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Train or run a seq2seq model to translate OOV words."))
    argparser.add_argument("--config_path", type=str,
                           help=("The path to the config file. The config file "
                                 "stores arguments that you may not want to type "
                                 "out every time. If an argument occurs in "
                                 "the config file and is passed in as a command-line "
                                 "flag, the value passed in through the command line "
                                 "will be used."))
    argparser.add_argument("--eval_path", type=str,
                           help=("The path to a source-target TSV file or a file with "
                                 "just source sequences (one per line) that we will use "
                                 "the loaded model to predict on"))
    argparser.add_argument("--eval_output_path", type=str,
                           help="The path to write eval predictions to. If not provided, "
                           "predictions will be written to a subfolder of the input eval "
                           "filepath called guess.")
    argparser.add_argument("--log_dir", type=str,
                           help=("The path of the folder to log train model "
                                 "progress to."))
    argparser.add_argument("--model_class", type=str,
                           help=("The name of the model class to "
                                 "instantiate."))
    argparser.add_argument("--model_params", type=json.loads,
                           help=("A dictionary of kwarg params"
                                 "(in json format) to be passed to the "
                                 "model constructor."))
    argparser.add_argument("--predict_from_path", type=str,
                           help=("The path to the checkpoint to load and predict with. "
                                 "We will also search for a config json file in the "
                                 "same directory to initialize the model with."))
    argparser.add_argument("--predict_interactive", type=bool,
                           help=("Whether or not to predict interactively from "
                                 "user input."))
    argparser.add_argument("--predict_params", type=json.loads,
                           help=("A dictionary of kwarg params"
                                 "(in json format) to be passed to the "
                                 "translate_file function."))
    argparser.add_argument("--read_data_params", type=json.loads,
                           help=("A dictionary of kwarg params"
                                 "(in json format) to be passed to the "
                                 "read_data function."))
    argparser.add_argument("--run_id", type=str,
                           help=("An identifying string for this run of "
                                 "the model."))
    argparser.add_argument("--save_dir", type=str,
                           help=("The path of the folder to save model "
                                 "checkpoints to."))
    argparser.add_argument("--n_jobs", type=int,
                           help=("The number of processes to use for "
                                 "translating, if multiprocessing is "
                                 "supported by the solver. Default is 1."))
    argparser.add_argument("--train_from_path", type=str,
                           help=("The path to the checkpoint to load and continue "
                                 "train with. We will also search for a config json "
                                 "file in the same directory to initialize the model "
                                 "with."))
    argparser.add_argument("--train_model_params", type=json.loads,
                           help=("A dictionary of kwarg params"
                                 "(in json format) to be passed to the "
                                 "train_model function."))

    if (os.path.abspath(os.getcwd()) !=
            os.path.abspath(os.path.join(os.path.dirname(__file__),
                                         os.pardir, os.pardir))):
        logger.warning("It's recommended to run this script from the project "
                       "root folder (python scripts/run/run_oov_solver.py ...), "
                       "since many of the experiment configs use relative paths. "
                       "Don't be surprised if things break.\n"
                       "Current directory: {}\n"
                       "Expected directory: {}".format(
                           os.path.abspath(os.getcwd()),
                           os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                        os.pardir, os.pardir))))

    input_args = vars(argparser.parse_args())
    config_path = input_args.pop("config_path")
    if config_path:
        with open(config_path) as config_file:
            config_from_file = json.load(config_file)
    elif (input_args.get("predict_from_path", None) or
          input_args.get("train_from_path", None)):
        # Find the original config used to construct the model
        if input_args.get("predict_from_path", None):
            load_folder = os.path.dirname(input_args["predict_from_path"])
        else:
            load_folder = os.path.dirname(input_args["train_from_path"])
        logger.info("Loading original config used to construct saved "
                    "model from {}".format(load_folder))
        config_files = [os.path.join(load_folder, f) for f in
                        os.listdir(load_folder) if f.endswith('config.json')]
        if len(config_files) != 1:
            raise ValueError("Found config files {}. Please ensure that you "
                             "have exactly one config file in the same "
                             "directory as the model you wish to "
                             "load.".format(config_files))
        original_config_path = config_files[0]
        with open(original_config_path) as original_config_file:
            config_from_file = json.load(original_config_file)
    else:
        config_from_file = {}

    # Combine the configs read from the file and the configs
    # passed in from the command line.
    config = deepcopy(config_from_file)
    for key in input_args:
        if input_args[key] is not None:
            if key in config_from_file:
                logger.info(
                    "Arg {} has value {} from config file and {} "
                    "from command line flag, using value from "
                    "command line flag.".format(key, config_from_file[key],
                                                input_args[key]))
            config[key] = input_args[key]

    if "model_class" not in config:
        raise ValueError("Value for argument \"model_class\" must be "
                         "provided in either an input config file or as a "
                         "command line argument.")

    # Get the class referring to the selected model
    model_class_str = config.get("model_class")
    if model_class_str not in implemented_models:
        raise ValueError("{} is not a valid model class. "
                         "Possible choices are {}".format(
                             model_class_str, implemented_models))
    model_class = implemented_models[model_class_str]

    if "run_id" not in config:
        raise ValueError("Value for argument \"run_id\" must be "
                         "provided in either an input config file or as a "
                         "command line argument.")

    run_id = config.get("run_id")

    if "predict_from_path" in config:
        if "eval_path" not in config and "predict_interactive" not in config:
            raise ValueError(
                "If loading a model, value for argument \"eval_path\" "
                "must be provided in either an input config file or as a "
                "command line argument.")
        predict_from_path = config.get("predict_from_path")

        logger.info("Loading model checkpoint at {} "
                    "for prediction".format(predict_from_path))

        # Create an instance of the model
        model_params = config.get("model_params", {})
        model = model_class(**model_params)
        model.load_from_file(predict_from_path)
        logger.info("Successfully loaded model!")

        if "eval_path" in config:
            eval_path = config.get("eval_path")
            logger.info("Translating sequences from {}".format(eval_path))
            if config.get("n_jobs", None) is None:
                n_jobs = 1
            else:
                n_jobs = config["n_jobs"]

            predict_params = config.get("predict_params", {})
            translated_sequences = model.translate_file(
                eval_path, n_jobs=n_jobs, **predict_params)

            # Write the translated sequences out.
            if "eval_output_path" not in config:
                guess_folder_path = os.path.join(os.path.dirname(eval_path), "guess")
                if not os.path.exists(guess_folder_path):
                    os.makedirs(guess_folder_path)
                eval_predicted_path = os.path.join(
                    guess_folder_path, os.path.basename(eval_path) + "." +
                    run_id + ".guess")
            else:
                eval_predicted_path = config["eval_output_path"]
            logger.info("Writing predicted translations "
                        "to {}".format(eval_predicted_path))
            with open(eval_predicted_path, "w") as eval_guesses:
                for translated_sequence in translated_sequences:
                    eval_guesses.write("{}\n".format(translated_sequence))
        else:
            while True:
                predict_params = config.get("predict_params", {})
                to_translate = input("Enter a word to translate: ")
                logger.info("Translating \"{}\"...".format(to_translate))
                translated_word = model.translate_list(
                    [to_translate], debug=True, show_progbar=False, **predict_params)
                logger.info("Predicted translation: {}".format(translated_word))

    # Train a model
    else:
        # Instantiate the selected model
        model_params = config.get("model_params", {})

        model = model_class(**model_params)

        if "train_from_path" in config:
            train_from_path = config.get("train_from_path")
            logger.info("Loading model checkpoint at {} "
                        "for training".format(train_from_path))
            model.load_from_file(train_from_path)
            logger.info("Successfully loaded model!")

        # Read data for the model
        logger.info("Reading data for the model.")
        read_data_params = config.get("read_data_params", {})
        data = model.read_data(**read_data_params)

        # Train the model
        logger.info("Training the model.")
        train_model_params = config.get("train_model_params", {})

        log_dir = config.get("log_dir", None)
        if log_dir is None:
            logger.warning("Value for log_dir was not provided, the "
                           "train progress will not be logged to a file.")
        else:
            log_dir = os.path.join(log_dir, run_id)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            logger.info("Logging model progress to {}".format(log_dir))

        save_dir = config.get("save_dir", None)
        if save_dir is None:
            logger.warning("Value for save_dir was not provided, the "
                           "trained model and config will not be saved.")
        else:
            save_dir = os.path.join(save_dir, run_id)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            logger.info("Will save model and config to {}".format(save_dir))
            saved_config_path = os.path.join(save_dir, run_id + "_config.json")
            if os.path.exists(saved_config_path):
                logger.info("Config already exists, not saving.")
            else:
                with open(saved_config_path, "w") as saved_config_file:
                    json.dump(config, saved_config_file, indent=4)

        # Combine data and train model params for python 2 compatibility
        combined_data_train_model_params = deepcopy(train_model_params)
        combined_data_train_model_params.update(data)
        model.train_model(log_dir=log_dir, save_dir=save_dir,
                          run_id=run_id,
                          **combined_data_train_model_params)


if __name__ == "__main__":
    # Redirect things to stdout, since we want to keep stderr clean for
    # hyperparameter optimzation
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO, handlers=handlers)
    main()
