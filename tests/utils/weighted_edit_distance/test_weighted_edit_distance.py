# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from ...common.test_case import OOVTestCase

from oov.utils.weighted_edit_distance.weighted_edit_distance import (
    get_weighted_edit_distance)


class TestWeightedEditDistanceUtils(OOVTestCase):
    def test_get_weighted_edit_distance(self):
        list1 = ["Xayyarri", "hordoftoonnii", "Pireezdaantiin", "imaltootaa",
                 "beeksiste", "masriifi", "Hanbaan"]
        list2 = ["xayyara", "hordoftuu", "pirezidaantii", "imaltu",
                 "beksisu", "Masriif", "hamba"]
        edit_distances = get_weighted_edit_distance(
            list1, list2, lang_code1="orm", lang_code2="orm")
        assert edit_distances == [0.22, 0.6, 0.52, 0.5, 1.22, 0.1, 0.82]

        with self.assertRaises(ValueError):
            get_weighted_edit_distance(["list", "list2"], ["list0"])
