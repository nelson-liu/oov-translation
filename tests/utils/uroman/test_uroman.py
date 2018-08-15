# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from ...common.test_case import OOVTestCase

from oov.utils.uroman.uroman import uromanize_list


class TestUromanUtils(OOVTestCase):
    def test_uromanize_list(self):
        wordlist = ["hello", "this", "is", "a", "uroman", "test"]
        uromanized_wordlist = uromanize_list(wordlist, self.uroman_path)
        assert uromanized_wordlist == wordlist

        unicode_wordlist = ["verás", "cámara", "poniéndoselo", "perro"]
        uromanized_wordlist = uromanize_list(unicode_wordlist, self.uroman_path)
        expected_uromanized_wordlist = ["veras", "camara", "poniendoselo", "perro"]
        assert uromanized_wordlist == expected_uromanized_wordlist

        nonroman_wordlist = ["ወንጀለኞች", "በአነስተኛ", "ኃይሌ", "ዕዳ"]
        uromanized_wordlist = uromanize_list(nonroman_wordlist, self.uroman_path)
        expected_uromanized_wordlist = ["wanejalanyoche", "baanasetanyaa",
                                        "xaayelee", "edaa"]
        assert uromanized_wordlist == expected_uromanized_wordlist
