import argparse
from copy import deepcopy
import json
import logging
import os
import random
import shlex
from subprocess import check_output, Popen
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Train or run a model to translate OOV words."))
    argparser.add_argument("--config_path", type=str,
                           help=("The path to the config file. The config file "
                                 "stores arguments that you may not want to type "
                                 "out every time. If an argument occurs in "
                                 "the config file and is passed in as a command-line "
                                 "flag, the value passed in through the command line "
                                 "will be used."))
    argparser.add_argument("--train_file", type=str,
                           help=("The path to a source-target TSV train file."))
    argparser.add_argument("--val_file", type=str,
                           help=("The path to a source-target TSV val file."))
    argparser.add_argument("--debug", action="store_true",
                           help=("Show output of subprocesses as they are run, "
                                 "for the purpose of debugging"))
    argparser.add_argument("--gpu_indices", type=int, nargs="+",
                           help=("The indices of the GPUs to run jobs on."))
    argparser.add_argument("--log_dir", type=str,
                           help=("The path of the folder to log train model "
                                 "progress to."))
    argparser.add_argument("--model_class", type=str,
                           help=("The name of the model class to "
                                 "instantiate."))
    argparser.add_argument("--model_param_grid", type=json.loads,
                           help=("A dictionary of kwarg params and lists of "
                                 "values to run (in json format) to be "
                                 "passed to the model constructor."))
    argparser.add_argument("--models_per_gpu", type=int,
                           help=("The maximum number of models to put on one GPU."))
    argparser.add_argument("--num_configs_to_try", type=int,
                           help=("The number of hyperparameter "
                                 "configurations to sample from the grid."))
    argparser.add_argument("--run_id", type=str,
                           help=("An identifying string for this run of "
                                 "the model."))
    argparser.add_argument("--save_dir", type=str,
                           help=("The path of the folder to save model "
                                 "checkpoints to."))
    argparser.add_argument("--train_model_params", type=json.loads,
                           help=("A dictionary of kwarg params"
                                 "(in json format) to be passed to the "
                                 "train_model function."))

    input_args = vars(argparser.parse_args())
    config_path = input_args.pop("config_path")
    if config_path:
        with open(config_path) as config_file:
            config_from_file = json.load(config_file)
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

    if "run_id" not in config:
        raise ValueError("Value for argument \"run_id\" must be "
                         "provided in either an input config file or as a "
                         "command line argument.")

    run_id = config.get("run_id")

    # Train a model
    # Input validation to make sure the required arguments are either
    # provided in the config file or in the command line
    if "train_file" not in config or "val_file" not in config:
        raise ValueError("Value for arguments \"train_file\" and \"val file\" "
                         "must be provided in either an input config file "
                         "or as a command line argument.")
    train_file = config.get("train_file")
    val_file = config.get("val_file")

    # Take the model param grid, and sample from it to get a list of
    # parameter configs

    model_param_grid = config.get("model_param_grid")
    num_configs_to_try = config.get("num_configs_to_try")
    sampled_model_configs = []
    for i in range(num_configs_to_try):
        current_config = {}
        while current_config == {} or current_config in sampled_model_configs:
            for key, value in model_param_grid.items():
                if "+" in key:
                    split_values = random.choice(value)
                    current_config.update(dict(list(zip(key.split("+"),
                                                        split_values))))
                else:
                    current_config[key] = random.choice(value)
        sampled_model_configs.append(current_config)

    # turn the model configs into commands to be run
    run_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "run_oov_solver.py")
    cwd = os.getcwd()

    commands_to_run = []
    for model_config in sampled_model_configs:
        command = "python " + run_script_path
        # append on the read data params
        read_data_params = {
            "train_file": os.path.join(cwd, train_file),
            "val_file": os.path.join(cwd, val_file)}
        command += " " + "--read_data_params " + "'" + json.dumps(read_data_params) + "'"

        # append the log dir
        log_dir = config.get("log_dir", None)
        command += " " + "--log_dir " + os.path.join(cwd, log_dir)

        # append the model class
        model_class = config.get("model_class")
        command += " " + "--model_class " + model_class

        # append the model params
        command += " " + "--model_params " + "'" + json.dumps(model_config) + "'"

        # append the run id
        process_run_id = run_id + "_" + json.dumps(model_config) \
                                            .replace("\"", "") \
                                            .replace("{", "") \
                                            .replace("}", "") \
                                            .replace(" ", "") \
                                            .replace(",", "_") \
                                            .replace(":", ".")
        command += " " + "--run_id " + process_run_id

        # append the save dir
        save_dir = config.get("save_dir")
        command += " " + "--save_dir " + os.path.join(cwd, save_dir)

        # append the train model params
        train_model_params = config.get("train_model_params", {})
        train_model_params["show_progbar"] = False
        command += (" " + "--train_model_params " + "'" +
                    json.dumps(train_model_params) + "'")
        commands_to_run.append(command)

    # Iterate through the commands_to_run, popping them off and running
    # them until there are none left.
    models_per_gpu = config.get("models_per_gpu", 3)
    gpu_indices = config.get("gpu_indices")
    command_count = 1
    while commands_to_run != []:
        for gpu_index in gpu_indices:
            while len(get_gpu_index_pids(gpu_index)) < models_per_gpu:
                # Launch a command on this gpu
                command_to_run = commands_to_run.pop()
                logger.info("About to start command {}/{}".format(
                    command_count, num_configs_to_try))

                current_env = os.environ.copy()
                current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                logger.info("CUDA_VISIBLE_DEVICES {}".format(
                    current_env["CUDA_VISIBLE_DEVICES"]))
                logger.info("Running command {}".format(shlex.split(command_to_run)))
                with open(os.devnull, 'w') as devnull:
                    if config["debug"]:
                        process = Popen(
                            shlex.split(command_to_run), env=current_env)
                    else:
                        process = Popen(
                            shlex.split(command_to_run), stdin=devnull,
                            stdout=devnull, close_fds=True,
                            env=current_env)
                    pid = process.pid
                    logger.info("Started with PID {}".format(pid))
                command_count += 1
                logger.info("Waiting to finish reading data "
                            "before starting next.")
                while pid not in get_gpu_index_pids(gpu_index):
                    time.sleep(30)


def get_gpu_index_pids(gpu_index):
    assert isinstance(gpu_index, int)
    # get a mapping from gpu index to uuid
    index_to_uuid_gpu_query_columns = ("index", "uuid")
    index_to_uuid_output = execute_process(
        r"nvidia-smi --query-gpu={query_cols} "
        "--format=csv,noheader,nounits".format(
            query_cols=",".join(index_to_uuid_gpu_query_columns)))
    index_to_uuid = {}
    for line in index_to_uuid_output.split("\n"):
        if not line:
            continue
        index, uuid = line.split(", ")
        index_to_uuid[int(index)] = uuid

    pid_query_columns = ("gpu_uuid", "pid")
    pid_output = execute_process(
        r"nvidia-smi --query-compute-apps={query_cols} "
        "--format=csv,noheader,nounits".format(
            query_cols=",".join(pid_query_columns)))
    index_pids = []
    for line in pid_output.split("\n"):
        if not line:
            continue
        uuid, pid = line.split(", ")
        if uuid == index_to_uuid[int(gpu_index)]:
            index_pids.append(int(pid))
    return index_pids


def execute_process(command_shell):
    stdout = check_output(command_shell, shell=True).strip()
    if not isinstance(stdout, (str)):
        stdout = stdout.decode()
    return stdout


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
