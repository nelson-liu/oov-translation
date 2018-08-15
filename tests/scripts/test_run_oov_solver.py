import os
import logging
import shutil
import subprocess

from ..common.test_case import OOVTestCase

from overrides import overrides

logger = logging.getLogger(__name__)


class TestRunOOVSolver(OOVTestCase):
    @overrides
    def setUp(self):
        super(TestRunOOVSolver, self).setUp()
        self.project_root_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)

    def test_run_char_seq2seq_end_to_end(self):
        # Change directory to project root with context manager
        with cd(self.project_root_path):
            run_oov_solver_path = os.path.join("scripts", "run", "run_oov_solver.py")
            config_path = os.path.join(
                "tests", "scripts", "configs",
                "test_run_oov_solver_char_seq2seq_config.json")
            # Create the save and log dir
            save_dir = os.path.join(
                self.test_dir,
                "unittest_saved_models/seq2seq/char")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            log_dir = os.path.join(
                self.test_dir,
                "unittest_saved_logs/seq2seq/char")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Train the char seq2seq oov solver.
            cmd = ["python", run_oov_solver_path, "--config_path", config_path,
                   "--save_dir", save_dir, "--log_dir", log_dir]
            logger.info("Running command: {}".format(cmd))
            child = subprocess.Popen(cmd)
            child.communicate()
            return_code = child.returncode
            if return_code != 0:
                print("Got return code {} when running char "
                      "seq2seq script".format(return_code))
                assert False
            # Test that loading a model and training more works
            # NOTE: This path is hardcoded, and can change if the underlying
            # data does.
            checkpoint_path = os.path.join(
                save_dir, "seq2seq_char_amh_unittest",
                "seq2seq_char_amh_unittest-step-25.ckpt")
            cmd = ["python", run_oov_solver_path,
                   "--train_from_path", checkpoint_path]
            child = subprocess.Popen(cmd)
            child.communicate()
            return_code = child.returncode
            if return_code != 0:
                print("Got return code {} when running char "
                      "seq2seq script".format(return_code))
                assert False

            # Test that loading a model and predicting works
            # NOTE: This path is hardcoded, and can change if the underlying
            # data does.
            checkpoint_path = os.path.join(
                save_dir, "seq2seq_char_amh_unittest",
                "seq2seq_char_amh_unittest-step-50.ckpt")
            eval_file_path = os.path.join(
                "tests", "test_data", "word_translation",
                "amh-eng.word_translation.test.sample")
            cmd = ["python", run_oov_solver_path, "--eval_path",
                   eval_file_path, "--predict_from_path", checkpoint_path]
            child = subprocess.Popen(cmd)
            child.communicate()
            # Remove test guess folder if it exists
            guess_folder = os.path.join(
                "tests", "test_data", "word_translation", "guess")
            if os.path.exists(guess_folder):
                try:
                    shutil.rmtree(guess_folder)
                except:
                    subprocess.call(["rm", "-rf", guess_folder])
            return_code = child.returncode
            if return_code != 0:
                print("Got return code {} when running char "
                      "seq2seq script".format(return_code))
                assert False


class cd:
    """Context manager for changing the current working directory
    From: https://stackoverflow.com/a/13197763/2544124
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        print("Entering {} from {}".format(self.newPath, self.savedPath))
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        print("Returning to {} from {}".format(self.savedPath, self.newPath))
        os.chdir(self.savedPath)
