import torch
import os

from ...common.test_case import OOVTestCase

from oov.models.similarity.edit_distance_solver import (
    EditDistanceSolver)
from overrides import overrides


class TestEditDistanceSolver(OOVTestCase):
    @overrides
    def setUp(self):
        super(TestEditDistanceSolver, self).setUp()

    def test_read_data(self):
        self.write_language_pair_data()
        # Use the model to read data
        model = EditDistanceSolver()
        tgt_given_src_path = os.path.join(
            self.language_pair_data_dir,
            "tgt_given_src.unnormalized")
        lexicon_path = os.path.join(
            self.language_pair_data_dir,
            "lexicon")
        read_data = model.read_data(tgt_given_src_path, lexicon_path)

        read_tgt_given_src_unnormalized = read_data["foreign_to_english"]
        assert isinstance(read_tgt_given_src_unnormalized, dict)
        assert read_tgt_given_src_unnormalized == {
            "somesource1": {"sometarget1": 7, "sometarget2": 3},
            "somesource3": {"sometarget2": 1},
            "amhword1close": {"amhword1translation1-1": 80,
                              "amhword1translation1-2": 10,
                              "amhword1translation1-3": 10},
            "amhword1further": {"amhword1translation-1": 4,
                                "amhword1translation-2": 6},
            "amhword2close": {"amhword2translation-1": 10},
            "amhword2clsr": {"amhword2translation-1": 45,
                             "amhword2translation-2": 55},
            "amhword2veryfurther": {"amhword2translation-3": 10},
            "amhword1notoov": {"amhword1 not oov translation": 10000},
            "amhword2notoov": {"amhword2notoov translation": 10000},
            "amhword3notoov": {"amhword3notoovtranslation": 10000}
        }

    def test_train_and_predict_with_original_format(self):
        self.write_language_pair_data()
        model = EditDistanceSolver()
        roman_model = EditDistanceSolver(uroman_path=True)
        tgt_given_src_path = os.path.join(
            self.language_pair_data_dir,
            "tgt_given_src.unnormalized")
        read_data = model.read_data(tgt_given_src_path)
        roman_read_data = roman_model.read_data(tgt_given_src_path)

        model.train_model(log_dir=self.language_pair_log_dir,
                          save_dir=self.language_pair_save_dir,
                          run_id="edit_distance_unit_test",
                          **read_data)
        roman_model.train_model(log_dir=self.language_pair_log_dir,
                                save_dir=self.language_pair_save_dir,
                                run_id="roman_edit_distance_unit_test",
                                **roman_read_data)

        amh_eng_dev = os.path.join(self.language_pair_data_dir, "dev")
        dev_translated = model.translate_file(amh_eng_dev, show_progbar=False)
        roman_dev_translated = roman_model.translate_file(amh_eng_dev, show_progbar=False)
        assert dev_translated == ["amhword1translation1-1",
                                  "amhword2translation-2"]
        assert roman_dev_translated == ["amhword1translation1-1",
                                        "amhword2translation-2"]
        # Load the model that was saved
        # Reconstruct the model with its state dict
        saved_model_path = os.path.join(
            self.language_pair_save_dir,
            "edit_distance_unit_test_model.pkl")
        state_dict = torch.load(saved_model_path)
        loaded_model = state_dict["solver_class"](**state_dict["solver_init_params"])
        loaded_model.load_from_file(saved_model_path)
        assert loaded_model.get_state_dict() == model.get_state_dict()
        loaded_dev_translated = loaded_model.translate_file(amh_eng_dev,
                                                            show_progbar=False)
        assert loaded_dev_translated == dev_translated

        # Reconstruct the roman model with its state dict
        saved_roman_model_path = os.path.join(
            self.language_pair_save_dir,
            "roman_edit_distance_unit_test_model.pkl")
        roman_state_dict = torch.load(saved_roman_model_path)
        loaded_roman_model = roman_state_dict["solver_class"](
            **roman_state_dict["solver_init_params"])
        loaded_roman_model.load_from_file(saved_roman_model_path)
        assert loaded_roman_model.get_state_dict() == roman_model.get_state_dict()
        loaded_roman_dev_translated = loaded_roman_model.translate_file(
            amh_eng_dev, show_progbar=False)
        assert loaded_roman_dev_translated == roman_dev_translated

    def test_translate_options(self):
        self.write_language_pair_data()
        model = EditDistanceSolver()
        tgt_given_src_path = os.path.join(
            self.language_pair_data_dir,
            "tgt_given_src.unnormalized")
        read_data = model.read_data(tgt_given_src_path)

        model.train_model(log_dir=self.language_pair_log_dir,
                          save_dir=self.language_pair_save_dir,
                          run_id="edit_distance_unit_test",
                          **read_data)
        amh_eng_dev = os.path.join(self.language_pair_data_dir, "dev")
        assert model.translate_file(
            amh_eng_dev, show_progbar=True, n_jobs=1) == [
                "amhword1translation1-1", "amhword2translation-2"]
        assert model.translate_file(
            amh_eng_dev, show_progbar=True, n_jobs=1, edit_distance="substring") == [
                "amhword1translation1-1", "amhword2translation-2"]

        assert model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=1) == [
                "amhword1translation1-1", "amhword2translation-2"]
        assert model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=1, edit_distance="weighted") == [
                "amhword1translation1-1", "amhword2translation-1"]
        assert model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=1, edit_distance="substring") == [
                "amhword1translation1-1", "amhword2translation-2"]

        assert model.translate_file(
            amh_eng_dev, show_progbar=True, n_jobs=2) == [
                "amhword1translation1-1", "amhword2translation-2"]
        assert model.translate_file(
            amh_eng_dev, show_progbar=True, n_jobs=2, edit_distance="substring") == [
                "amhword1translation1-1", "amhword2translation-2"]

        assert model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=2) == [
                "amhword1translation1-1", "amhword2translation-2"]
        assert model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=2, edit_distance="substring") == [
                "amhword1translation1-1", "amhword2translation-2"]
        assert model.translate_file(
            amh_eng_dev, show_progbar=False, n_jobs=2, edit_distance="weighted") == [
                "amhword1translation1-1", "amhword2translation-1"]

        list_to_translate = ["amhword1", "amhword2"]
        assert model.translate_list(
            list_to_translate, show_progbar=False, debug=True, n_jobs=1) == [
                "amhword1translation1-1", "amhword2translation-2"]
        assert model.translate_list(
            list_to_translate, show_progbar=False, debug=False, n_jobs=1) == [
                "amhword1translation1-1", "amhword2translation-2"]
