from .plural.plural_string_solver import PluralStringSolver
from .seq2seq.char_seq2seq_translation_model import CharSeq2SeqTranslationModel
from .seq2seq.char_trigram_seq2seq_translation_model import (
    CharTrigramSeq2SeqTranslationModel)
from .seq2seq.morph2char_translation_model import (
    Morph2CharTranslationModel)
from oov.models.seq2seq.morph_seq2seq_translation_model import (
    MorphSeq2SeqTranslationModel)
from .seq2seq.seq2seq_ensemble_translation_model import (
    Seq2SeqEnsembleTranslationModel)
from .seq2seq.multilingual_char_seq2seq_translation_model import (
    MultilingualCharSeq2SeqTranslationModel)
from .seq2seq.word_seq2seq_translation_model import WordSeq2SeqTranslationModel
from .similarity.edit_distance_solver import (
    EditDistanceSolver)
from .similarity.edit_vector_combined_distance_solver import (
    EditVectorCombinedDistanceSolver)
from .similarity.vector_distance_solver import (
    VectorDistanceSolver)

implemented_models = {
    "CharSeq2SeqTranslationModel": CharSeq2SeqTranslationModel,
    "CharTrigrmSeq2SeqTranslationModel": CharTrigramSeq2SeqTranslationModel,
    "EditDistanceSolver": EditDistanceSolver,
    "EditVectorCombinedDistanceSolver": EditVectorCombinedDistanceSolver,
    "Morph2CharTranslationModel": Morph2CharTranslationModel,
    "MorphSeq2SeqTranslationModel": MorphSeq2SeqTranslationModel,
    "MultilingualCharSeq2SeqTranslationModel": MultilingualCharSeq2SeqTranslationModel,
    "PluralStringSolver": PluralStringSolver,
    "Seq2SeqEnsembleTranslationModel": Seq2SeqEnsembleTranslationModel,
    "WordSeq2SeqTranslationModel": WordSeq2SeqTranslationModel,
    "VectorDistanceSolver": VectorDistanceSolver
}
