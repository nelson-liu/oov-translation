import torch
import os

from ...common.test_case import OOVTestCase

from oov.models.seq2seq.morph_seq2seq_translation_model import (
    MorphSeq2SeqTranslationModel)
from overrides import overrides


class TestMorphSeq2SeqTranslationModel(OOVTestCase):
    @overrides
    def setUp(self):
        super(TestMorphSeq2SeqTranslationModel, self).setUp()
        self.batch_size = 3
        self.embed_dim = 4
        self.rnn_dim = 6
        self.num_layers = 2
        self.output_projection_batch_size = 2
        self.model = MorphSeq2SeqTranslationModel(
            batch_size=self.batch_size,
            embed_dim=self.embed_dim,
            rnn_dim=self.rnn_dim,
            output_projection_batch_size=self.output_projection_batch_size,
            num_layers=self.num_layers,
            lr_decay=0.9,
            start_decay_at=4)

    def test_morpheme_segmentation(self):
        self.write_word_translation_data()
        self.model.read_data(
            self.spa_word_translation_train,
            self.spa_word_translation_val,
            self.spa_word_translation_src_segs,
            self.spa_word_translation_tgt_segs)
        assert self.model.morphemes_split_src_word("perritos") == ["perrito", "s"]
        assert self.model.morphemes_split_src_word("oov") == ["o", "o", "v"]
        assert self.model.morphemes_split_src_word("oov 1") == ["o", "o", "v", " ", "1"]
        assert (self.model.morphemes_split_src_word("test gato") ==
                ["t", "e", "s", "t", " ", "gato"])
        assert (self.model.morphemes_split_src_word("perritos amarillas") ==
                ["perrito", "s", " ", "amarill", "a", "s"])

        assert self.model.morphemes_split_tgt_word("dogs") == ["dog", "s"]
        assert self.model.morphemes_split_tgt_word("oov") == ["o", "o", "v"]
        assert self.model.morphemes_split_tgt_word("oov 1") == ["o", "o", "v", " ", "1"]
        assert (self.model.morphemes_split_tgt_word("yellow cats") ==
                ["yellow", " ", "cat", "s"])

    def test_train_and_load_gives_identical_results(self):
        num_epochs = 5
        self.write_word_translation_data()
        read_data = self.model.read_data(
            self.spa_word_translation_train,
            self.spa_word_translation_val,
            self.spa_word_translation_src_segs,
            self.spa_word_translation_tgt_segs)

        self.model.train_model(save_dir=self.language_pair_word_translation_save_dir,
                               log_dir=self.language_pair_word_translation_log_dir,
                               log_period=3, val_period=3, max_to_keep=0,
                               run_id="morph_seq2seq_unit_test", show_progbar=False,
                               num_epochs=num_epochs, **read_data)
        translate_test = ["amarillas", "word", "another word"]
        predicted_translations = self.model.translate_list(
            translate_test, show_progbar=False)
        predicted_val_translations = self.model.translate_file(
            self.spa_word_translation_val, show_progbar=False)

        last_checkpoint_path = os.path.join(
            self.language_pair_word_translation_save_dir,
            "morph_seq2seq_unit_test-step-{}.ckpt".format(self.model.global_step))
        state_dict = torch.load(open(last_checkpoint_path, "rb"))
        loaded_model = state_dict["solver_class"](**state_dict["solver_init_params"])
        loaded_model.load_from_file(last_checkpoint_path)
        # Check that the loaded model and trained model have identical parameters.
        for model_params, loaded_params in zip(self.model.model.parameters(),
                                               loaded_model.model.parameters()):
            if not torch.equal(model_params.data, loaded_params.data):
                assert torch.equal(model_params.data.view(-1),
                                   loaded_params.data.view(-1))

        # Check that the loaded model and trained model make the same predictions.
        loaded_model_predicted_translations = loaded_model.translate_list(
            translate_test, show_progbar=False)
        assert predicted_translations == loaded_model_predicted_translations
        loaded_model_predicted_val_translations = loaded_model.translate_file(
            self.spa_word_translation_val, show_progbar=False)
        assert predicted_val_translations == loaded_model_predicted_val_translations

    def test_training_after_loading(self):
        num_epochs = 5
        self.write_word_translation_data()
        read_data = self.model.read_data(
            self.spa_word_translation_train,
            self.spa_word_translation_val,
            self.spa_word_translation_src_segs,
            self.spa_word_translation_tgt_segs)
        self.model.train_model(save_dir=self.language_pair_word_translation_save_dir,
                               log_dir=self.language_pair_word_translation_log_dir,
                               log_period=3, val_period=3, max_to_keep=0,
                               run_id="morph_seq2seq_unit_test",
                               show_progbar=False, num_epochs=num_epochs, **read_data)
        last_checkpoint_path = os.path.join(
            self.language_pair_word_translation_save_dir,
            "morph_seq2seq_unit_test-step-{}.ckpt".format(self.model.global_step))
        state_dict = torch.load(open(last_checkpoint_path, "rb"))
        loaded_model = state_dict["solver_class"](**state_dict["solver_init_params"])
        loaded_model.load_from_file(last_checkpoint_path)
        # Check that the loaded model and trained model have identical parameters.
        for model_params, loaded_params in zip(self.model.model.parameters(),
                                               loaded_model.model.parameters()):
            if not torch.equal(model_params.data, loaded_params.data):
                assert torch.equal(model_params.data.view(-1),
                                   loaded_params.data.view(-1))

        # Train some more with the loaded model
        num_epochs = 2
        loaded_model.train_model(
            save_dir=self.language_pair_word_translation_save_dir,
            log_dir=self.language_pair_word_translation_log_dir,
            log_period=3, val_period=3, run_id="morph_seq2seq_unit_test",
            num_epochs=num_epochs, max_to_keep=0, show_progbar=False, **read_data)
        for model_params, loaded_params in zip(self.model.model.parameters(),
                                               loaded_model.model.parameters()):
            assert not torch.equal(model_params.data.view(-1),
                                   loaded_params.data.view(-1))
