# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import torch
import os

from ...common.test_case import OOVTestCase

from oov.models.plural.plural_string_solver import (
    PluralStringSolver)
from overrides import overrides


class TestPluralStringSolver(OOVTestCase):
    @overrides
    def setUp(self):
        super(TestPluralStringSolver, self).setUp()
        self.model = PluralStringSolver()

    def test_get_singularization_transforms(self):
        singular_plural_pairs = [
            ("hostería", "hosterías"),
            ("decapita", "decapitaciones"),
            ("díario", "diarios"),
            ("craneo", "cráneos"),
            ("suposición", "suposiciones"),
            ("entrenadora", "entrenadores"),
            ("simpatisante", "simpatizantes"),
            ("pintor", "pintores"),
            ("oficiosa", "oficiales"),
            ("compañia", "compañias"),
            ("imaginación", "imaginaciones"),
            ("cardiólogo", "cardiólogos")]
        sample_transforms, all_transforms = self.model._get_singularization_transforms(
            singular_plural_pairs)
        assert all_transforms == set([
            frozenset({"-s$"}),
            frozenset({"-ciones$"}),
            frozenset({"-s$", "-^di", "+^dí"}),
            frozenset({"-s$", "+^cra", "-^crá"}),
            frozenset({"-ones$", "+ón$"}),
            frozenset({"-es$", "+a$"}),
            frozenset({"-zantes$", "+sante$"}),
            frozenset({"-es$"}),
            frozenset({"-ales$", "+osa$"})])
        assert sample_transforms == [
            {'-s$'}, {'-ciones$'}, {'-s$', '+^dí', '-^di'},
            {'-s$', '+^cra', '-^crá'}, {'-ones$', '+ón$'},
            {'-es$', '+a$'}, {'+sante$', '-zantes$'}, {'-es$'},
            {'+osa$', '-ales$'}, {'-s$'}, {'-ones$', '+ón$'}, {'-s$'}]

    def test_apply_transforms(self):
        singular_plural_pairs = [
            ("hostería", "hosterías"),
            ("decapita", "decapitaciones"),
            ("díario", "diarios"),
            ("craneo", "cráneos"),
            ("suposición", "suposiciones"),
            ("entrenadora", "entrenadores"),
            ("simpatisante", "simpatizantes"),
            ("pintor", "pintores"),
            ("oficiosa", "oficiales"),
            ("compañia", "compañias"),
            ("imaginación", "imaginaciones"),
            ("cardiólogo", "cardiólogos")]
        sample_transforms, all_transforms = self.model._get_singularization_transforms(
            singular_plural_pairs)
        for singular_plural_pair, transforms in zip(singular_plural_pairs,
                                                    sample_transforms):
            singular, plural = singular_plural_pair
            transformed_plural = self.model._apply_transforms(plural, transforms)
            assert transformed_plural == singular

    def test_read_data_train_model_translate(self):
        # Write data, and use model to read it.
        self.write_plural_solver_data()
        data = self.model.read_data(
            self.plural_solver_en_fr_sing_plural_pairs_train,
            self.plural_solver_tgt_given_src_path,
            self.plural_solver_en_fr_sing_plural_pairs_val)

        # Train the model.
        self.model.train_model(
            save_dir=self.plural_solver_save_dir,
            run_id="plural_string_solver_unit_test",
            **data)

        # Use trained model to get translations.
        original_translations = self.model.translate_list(
            ["kioscos", "proteínas", "variaciones"])
        original_val_translations = self.model.translate_file(
            self.plural_solver_en_fr_sing_plural_pairs_val)

        # Reconstruct the model with its state dict from disk.
        state_dict = torch.load(os.path.join(
            self.plural_solver_save_dir,
            "plural_string_solver_unit_test_model.pkl"))
        # Load model state from disk.
        loaded_model = state_dict["solver_class"](**state_dict["solver_init_params"])
        loaded_model = loaded_model.load_from_file(os.path.join(
            self.plural_solver_save_dir,
            "plural_string_solver_unit_test_model.pkl"))

        # Assert that training a loaded plural_string_solver raises error.
        with self.assertRaises(ValueError):
            loaded_model.train_model(**data)

        # Translate with the loaded plural_string_solver.
        loaded_translations = loaded_model.translate_list(
            ["kioscos", "proteínas", "variaciones"])
        loaded_val_translations = loaded_model.translate_file(
            self.plural_solver_en_fr_sing_plural_pairs_val)

        # Assert that the translations between the original and loaded
        # solvers are identical.
        assert original_translations == loaded_translations
        assert original_val_translations == loaded_val_translations
