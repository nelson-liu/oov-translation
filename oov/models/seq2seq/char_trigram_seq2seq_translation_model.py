from overrides import overrides
from torchtext import data

from .base_seq2seq_translation_model import BaseSeq2SeqTranslationModel


class CharTrigramSeq2SeqTranslationModel(BaseSeq2SeqTranslationModel):
    """
    This model is a sequence-to-sequence model for translating between languages
    that operates on character trigrams, largely inspired by the positive results
    reported in "From Characters to Words to in Between: Do We Capture Morphology?" by
    Clara Vania and Adam Lopez (ACL 2017, https://arxiv.org/abs/1704.08352).

    To train, we pass the model a sequence of character trigrams in the source language
    and the model is tasked with predicting a sequence of character trigrams in the target
    language.

    At test time, we are shown an OOV word to translate as simply a sequence
    of character trigrams, and we accordingly predict its target translation as
    a sequence of character trigrams.
    """

    @overrides
    def read_data(self, train_file, val_file, max_source_vocab=50000,
                  max_target_vocab=50000):
        if self.source_field is None:
            self.source_field = data.Field(tokenize=self.character_trigrams_split_word,
                                           include_lengths=True)
        if self.target_field is None:
            self.target_field = data.Field(tokenize=self.character_trigrams_split_word,
                                           include_lengths=True,
                                           init_token="<bos>",
                                           eos_token="<eos>")
        return super(CharTrigramSeq2SeqTranslationModel, self).read_data(
            train_file=train_file,
            val_file=val_file,
            max_source_vocab=max_source_vocab,
            max_target_vocab=max_target_vocab)

    @staticmethod
    def character_trigrams_split_word(input_string):
        """
        Given an input string, segment it into character trigrams.

        For example, "wordss" is segmented as ['wor', 'ord', 'rds', 'dss']

        Parameters
        ----------
        input_string: str
            The string to break into character trigrams.

        Returns
        -------
        chunked_string: List of str
             A list of length 3 strings representing the character trigrams.
        """
        return list(map("".join, zip(*[input_string[i:] for i in range(3)])))

    @overrides
    def _format_output(self, output):
        return "".join(output)
