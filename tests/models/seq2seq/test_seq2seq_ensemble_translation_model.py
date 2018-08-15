import torch
import os

from ...common.test_case import OOVTestCase

from oov.models.seq2seq.char_seq2seq_translation_model import (
    CharSeq2SeqTranslationModel)
from oov.models.seq2seq.seq2seq_ensemble_translation_model import (
    Seq2SeqEnsembleTranslationModel)
from overrides import overrides


class TestSeq2SeqEnsembleTranslationModel(OOVTestCase):
    @overrides
    def setUp(self):
        super(TestSeq2SeqEnsembleTranslationModel, self).setUp()

    def test_train_and_load_gives_identical_results(self):
        self.write_word_translation_data()
        # Train 3 character seq2seq translations models with varying rnn dim
        models = {}
        for rnn_dim in [4, 6, 8]:
            run_id = "ensemble_unit_test_rnn_dim_{}".format(rnn_dim)
            model = CharSeq2SeqTranslationModel(
                batch_size=3, embed_dim=4, rnn_dim=rnn_dim,
                output_projection_batch_size=2, num_layers=2,
                lr_decay=0.9, start_decay_at=4)
            read_data = model.read_data(
                self.spa_word_translation_train,
                self.spa_word_translation_val)
            model.train_model(
                save_dir=self.language_pair_word_translation_save_dir,
                log_dir=self.language_pair_word_translation_log_dir,
                log_period=3, val_period=3, max_to_keep=None,
                run_id=run_id, num_epochs=5, show_progbar=False, **read_data)
            last_checkpoint_path = os.path.join(
                self.language_pair_word_translation_save_dir,
                "{}-step-{}.ckpt".format(
                    run_id, model.global_step))
            models[run_id] = {
                "model": model,
                "checkpoint_path": last_checkpoint_path}

        # Build an ensemble with the model objects
        object_ensemble = Seq2SeqEnsembleTranslationModel(
            {k: v["model"] for k, v in models.items()})
        # Make some predictions
        translate_test = ["test", "test2", "another word"]
        pred_object_ensemble_translations = object_ensemble.translate_list(
            translate_test, show_progbar=False)
        pred_object_ensemble_val_translations = object_ensemble.translate_file(
            self.spa_word_translation_val,
            show_progbar=False)

        # Build an ensemble with the checkpoint paths
        ckpt_ensemble = Seq2SeqEnsembleTranslationModel(
            {k: v["checkpoint_path"] for k, v in models.items()})
        # Make some predictions
        translate_test = ["test", "test2", "another word"]
        pred_ckpt_ensemble_translations = ckpt_ensemble.translate_list(
            translate_test, show_progbar=False)
        pred_ckpt_ensemble_val_translations = ckpt_ensemble.translate_file(
            self.spa_word_translation_val,
            show_progbar=False)

        assert (pred_object_ensemble_translations ==
                pred_ckpt_ensemble_translations)
        assert (pred_object_ensemble_val_translations ==
                pred_ckpt_ensemble_val_translations)

        # "Train" the object ensemble model to save it to disk
        object_ensemble.train_model(
            save_dir=self.language_pair_word_translation_save_dir,
            log_dir=self.language_pair_word_translation_log_dir,
            run_id="object_ensemble_serialized_unittest")
        # Reconstruct the object ensemble model
        saved_object_ensemble_path = os.path.join(
            self.language_pair_word_translation_save_dir,
            "object_ensemble_serialized_unittest_model.pkl")
        state_dict = torch.load(saved_object_ensemble_path)
        loaded_object_ensemble = state_dict["solver_class"](
            **state_dict["solver_init_params"])
        loaded_object_ensemble.load_from_file(saved_object_ensemble_path)
        pred_loaded_object_ensemble_translations = loaded_object_ensemble.translate_list(
            translate_test, show_progbar=False)
        pred_loaded_object_ensemble_val_translations = (
            loaded_object_ensemble.translate_file(
                self.spa_word_translation_val,
                show_progbar=False))
        assert (pred_object_ensemble_translations ==
                pred_loaded_object_ensemble_translations)
        assert (pred_object_ensemble_val_translations ==
                pred_loaded_object_ensemble_val_translations)

        # "Train" the ckpt ensemble model to save it to disk
        ckpt_ensemble.train_model(
            save_dir=self.language_pair_word_translation_save_dir,
            log_dir=self.language_pair_word_translation_log_dir,
            run_id="ckpt_ensemble_serialized_unittest")
        # Reconstruct the checkpoint ensemble model
        saved_ckpt_ensemble_path = os.path.join(
            self.language_pair_word_translation_save_dir,
            "ckpt_ensemble_serialized_unittest_model.pkl")
        state_dict = torch.load(saved_ckpt_ensemble_path)
        loaded_ckpt_ensemble = state_dict["solver_class"](
            **state_dict["solver_init_params"])
        loaded_ckpt_ensemble.load_from_file(saved_ckpt_ensemble_path)
        pred_loaded_ckpt_ensemble_translations = loaded_ckpt_ensemble.translate_list(
            translate_test, show_progbar=False)
        pred_loaded_ckpt_ensemble_val_translations = loaded_ckpt_ensemble.translate_file(
            self.spa_word_translation_val,
            show_progbar=False)
        assert (pred_ckpt_ensemble_translations ==
                pred_loaded_ckpt_ensemble_translations)
        assert (pred_ckpt_ensemble_val_translations ==
                pred_loaded_ckpt_ensemble_val_translations)
