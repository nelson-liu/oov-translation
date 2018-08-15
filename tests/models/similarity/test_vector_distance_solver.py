import torch
import json
import os
import subprocess

from numpy.testing import assert_allclose

from ...common.test_case import OOVTestCase
from ...common.test_markers import slow

from oov.models.similarity.vector_distance_solver import (
    VectorDistanceSolver)
from overrides import overrides


class TestVectorDistanceSolver(OOVTestCase):
    @overrides
    def setUp(self):
        super(TestVectorDistanceSolver, self).setUp()
        # Compile the FastText submodule
        fasttext_src_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.pardir, os.pardir, os.pardir, "util", "fastText")
        cmd = "make -C {}".format(fasttext_src_path)
        print("Running command {}".format(cmd))
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out, err = output.communicate()
        print("stdout:", out)
        print("stderr:", err)
        self.fasttext_binary_path = os.path.join(
            fasttext_src_path, "fasttext")
        self.model = VectorDistanceSolver()
        self.amh_fasttext_data = self.download_amh_fasttext()

    @slow
    def test_train_and_predict(self):
        self.write_language_pair_data()
        tgt_given_src_path = os.path.join(
            self.language_pair_data_dir,
            "tgt_given_src.unnormalized")
        read_data = self.model.read_data(
            tgt_given_src_path=tgt_given_src_path,
            **self.amh_fasttext_data)

        self.model.train_model(log_dir=self.language_pair_log_dir,
                               save_dir=self.language_pair_save_dir,
                               run_id="vector_distance_solver_unit_test",
                               **read_data)

        amh_eng_dev = os.path.join(self.language_pair_data_dir, "dev")
        dev_translated = self.model.translate_file(amh_eng_dev, show_progbar=False)
        assert dev_translated == ["amhword2translation-2",
                                  "amhword2translation-2"]

        # Reconstruct the model with its state dict
        saved_model_path = os.path.join(
            self.language_pair_save_dir,
            "vector_distance_solver_unit_test_model.pkl")
        state_dict = torch.load(saved_model_path)
        loaded_model = state_dict["solver_class"](**state_dict["solver_init_params"])
        loaded_model.load_from_file(saved_model_path)
        loaded_state_dict = loaded_model.get_state_dict()
        original_state_dict = self.model.get_state_dict()
        assert set(loaded_state_dict.keys()) == set(original_state_dict.keys())
        assert (set(loaded_state_dict["foreign_vectors"].keys()) ==
                set(original_state_dict["foreign_vectors"].keys()))
        for key in loaded_state_dict["foreign_vectors"]:
            assert_allclose(loaded_state_dict["foreign_vectors"][key],
                            original_state_dict["foreign_vectors"][key])

        assert (set(loaded_state_dict["foreign_to_english"].keys()) ==
                set(original_state_dict["foreign_to_english"].keys()))
        assert (json.dumps(loaded_state_dict["foreign_to_english"], sort_keys=True) ==
                json.dumps(original_state_dict["foreign_to_english"], sort_keys=True))
        loaded_dev_translated = loaded_model.translate_file(amh_eng_dev,
                                                            show_progbar=False)
        assert loaded_dev_translated == dev_translated

        # Test the various translation parameters
        assert loaded_model.translate_file(
            amh_eng_dev, show_progbar=True, n_jobs=1) == loaded_dev_translated
        assert loaded_model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=1) == loaded_dev_translated
        assert loaded_model.translate_file(
            amh_eng_dev, show_progbar=True, n_jobs=2) == loaded_dev_translated
        assert loaded_model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=2) == loaded_dev_translated
