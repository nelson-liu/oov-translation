from __future__ import division
import torch
import functools
import heapq
import logging
import multiprocessing
import os

import dill
import editdistance
from overrides import overrides
import six
from tqdm import tqdm

from ..base_solver import BaseSolver
from ...utils.general.data import get_num_lines
from ...utils.uroman.uroman import uromanize_list
from ...utils.weighted_edit_distance.weighted_edit_distance import (
    get_weighted_edit_distance)


logger = logging.getLogger(__name__)


def _get_nearest_source_mp_alias(edit_distance_solver, edit_distance,
                                 lang_code, oov_token):
    """
    Alias for EditDistanceSolver._get_nearest_source that allows the
    method to be called in a multiprocessing pool
    """
    return edit_distance_solver._get_nearest_source(
        oov_token, edit_distance=edit_distance, lang_code=lang_code)


class EditDistanceSolver(BaseSolver):
    """
    A simple baseline model for doing OOV translation that takes
    a linear interpolation of the similarity ratio
    (Levenshtein-based similarity measure) and the partial string
    similarity ratio (the score of the best matching substring of
    the longer sequence).

    We find the word with the highest similarity score in the source
    vocabulary, and pick its most likely translation (according to the
    t-table) as our predicted translation.

    Attributes
    ----------
    foreign_to_english: Dict of str to Dict of str
        A dict mapping foreign strings to dictionaries containing
        English translations as the keys and weights as the values

    Parameters
    ----------
    uroman_path: str, optional, default=None
        The path to the uroman executable. If None, no romanization is done.
        If True, the path is set to the copy of uroman included with this
        package. If a string, the path is set to that string (if it exists).
    """

    def __init__(self, uroman_path=None):
        self.solver_init_params = locals()
        self.solver_init_params.pop("self")
        self.solver_init_params.pop("__class__", None)

        # Set uroman path to the utils directory if it's not provided
        if uroman_path is True:
            uroman_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                os.pardir, "util", "uroman-v1.2", "bin", "uroman.pl"))
            self.solver_init_params["uroman_path"] = uroman_path

        if isinstance(uroman_path, six.string_types):
            if not os.path.exists(uroman_path):
                raise OSError(
                    "Input uroman executable path was {}, but the path does "
                    "not exist.".format(uroman_path))
            self.uroman_path = os.path.normpath(uroman_path)
        elif uroman_path is None:
            self.uroman_path = None
        else:
            raise ValueError("Expected None, True, or a string path for "
                             "uroman_path, got {}".format(uroman_path))

        self.foreign_to_english = None
        self.was_loaded = False

    @overrides
    def get_state_dict(self):
        state_dict = {
            "solver_class": self.__class__,
            "solver_init_params": self.solver_init_params,
            "uroman_path": self.uroman_path,
            "foreign_to_english": self.foreign_to_english,
            "token_counts": self.token_counts
        }
        return state_dict

    @overrides
    def load_from_state_dict(self, state_dict):
        self.foreign_to_english = state_dict["foreign_to_english"]
        self.uroman_path = state_dict["uroman_path"]
        self.token_counts = state_dict["token_counts"]
        self.was_loaded = True
        return self

    @overrides
    def load_from_file(self, load_path):
        state_dict = torch.load(load_path)
        return self.load_from_state_dict(state_dict)

    @overrides
    def read_data(self, tgt_given_src_path, lexicon_path=None,
                  counts_path=None):
        """
        Read the tgt_given_src t-table to get a mapping from foreign
        word to possible English translations.

        Parameters
        -----------
        tgt_given_src_path: str
            Path to t-table with probabilities of target (English)
            tokens given a source token. The expected format is:
            <english token><space><foreign token><space><weight>

        lexicon_path: str, optional (default=None)
            Path to the lexicon, which is a TSV file of
            <word>\t<POS>\t<translation>. We use this to augment the
            t-table.

        counts_path: str, optional (default=None)
            Path to a TSV file, where the first two columns are
            <word>\t<count>. Ties will be broken during translation
            by picking the word with the highest count.

        Returns
        -------
        data_dict: str
            A dictionary of kwargs to be passed to the train_model
            function.
        """
        if self.was_loaded:
            raise ValueError("EditDistanceSolver does not support "
                             "training from a saved model.")
        foreign_to_english = {}
        num_skipped = 0
        with open(tgt_given_src_path) as tgt_given_src_file:
            for line in tqdm(tgt_given_src_file,
                             total=get_num_lines(tgt_given_src_path)):
                split_line = line.rstrip("\n").split(" ")
                if len(split_line) == 3:
                    target, source, weight = split_line
                else:
                    num_skipped += 1
                    continue
                if source not in foreign_to_english:
                    foreign_to_english[source] = {}
                foreign_to_english[source][target] = float(weight)
            if num_skipped != 0:
                logger.warning("Skipped {} lines in {} due to improper "
                               "whitespace format.".format(
                                   num_skipped, tgt_given_src_path))

        if lexicon_path is not None:
            num_skipped = 0
            num_in_dictionary = 0
            num_added = 0
            with open(lexicon_path) as lexicon_file:
                for line in tqdm(lexicon_file,
                                 total=get_num_lines(lexicon_path)):
                    split_line = line.rstrip("\n").split("\t")
                    if len(split_line) == 3:
                        source, _, target = split_line
                    elif len(split_line) == 2:
                        source, target == split_line
                    else:
                        num_skipped += 1
                        continue
                    if source not in foreign_to_english:
                        foreign_to_english[source] = {}
                    else:
                        num_in_dictionary += 1
                    foreign_to_english[source][target] = float(10000)
                    num_added += 1
                if num_skipped != 0:
                    logger.warning("Skipped {} lines in {} due to improper "
                                   "whitespace format.".format(
                                       num_skipped, lexicon_path))
                logger.info("Added {} new entries from lexicon, "
                            "there are {} total entries".format(
                                (num_added - num_in_dictionary),
                                len(foreign_to_english)))

        token_counts = {}
        if counts_path is not None:
            logger.info("Reading token counts from {}".format(counts_path))
            with open(counts_path) as counts_file:
                for line in tqdm(counts_file,
                                 total=get_num_lines(counts_path)):
                    split_line = line.rstrip("\n").split("\t")[:2]
                    token, count = split_line
                    token_counts[token] = int(count)
            logger.warning("Read {} token counts".format(len(token_counts)))

        return {"foreign_to_english": foreign_to_english,
                "token_counts": token_counts}

    @overrides
    def save_to_file(self, save_dir, run_id):
        save_path = os.path.join(save_dir, run_id + "_model.pkl")
        state_dict = self.get_state_dict()
        torch.save(state_dict, save_path, pickle_module=dill)

    @overrides
    def train_model(self, foreign_to_english, token_counts, log_dir=None,
                    save_dir=None, run_id=None):
        if self.was_loaded:
            raise ValueError("EditDistanceSolver does not support "
                             "training from a saved model.")
        if self.uroman_path is not None:
            logger.info("Romanizing the t-table with uroman "
                        "at {}".format(self.uroman_path))
            # Get all of the source words in the t-table, and uromanize them.
            source_words = list(foreign_to_english.keys())
            romanized_source_words = uromanize_list(
                source_words, self.uroman_path)
            self.foreign_to_english = {
                roman_source_word: foreign_to_english[source_word] for
                source_word, roman_source_word in
                zip(source_words, romanized_source_words)}
            logger.info("Romanizing the token counts with uroman "
                        "at {}".format(self.uroman_path))
            # TODO: uromanize token counts as well?
            self.token_counts = token_counts
        else:
            self.foreign_to_english = foreign_to_english
            self.token_counts = token_counts
        logger.info("T-table has {} source entries".format(len(self.foreign_to_english)))

        if save_dir is not None and run_id is not None:
            logger.info("Saving trained model to save dir {} with run "
                        "id {}".format(save_dir, run_id))
            self.save_to_file(save_dir=save_dir, run_id=run_id)

    @overrides
    def translate_file(self, oov_path, show_progbar=True, n_jobs=1,
                       edit_distance="vanilla", lang_code=None):
        """
        Given a file, predict translations for each data example
        (line of file).

        Parameters
        ----------
        oov_path: str
            Path to file with data. The BaseSeq2SeqTranslationModel takes
            two kinds of data --- either vanilla OOV data as in the
            original dataset dev and test splits, OOV data as generated
            by the create_word_translation_data script, or a file containing
            just foreign OOVs to translate.

        edit_distance: str, optional (default=vanilla)
            The edit distance variant to use, one of "vanilla"
            (normal edit distance), "substring" (edit distance with higher weight on
            substrings) or "weighted" (edit distance with varying weights for different
            substitutions based on linguistic plausibility).

        lang_code: str
            The language code to pass to the weighted edit distance, if applicable.
        """
        sequences_to_translate = []
        num_cols = None
        with open(oov_path) as src_file:
            for line in src_file:
                split_line = line.rstrip("\n").split("\t")
                if num_cols is None:
                    num_cols = len(split_line)
                else:
                    assert num_cols == len(split_line)
                # Sequence to translate depends on num columns
                if len(split_line) == 1:
                    # File of just OOVs
                    sequences_to_translate.append(split_line[0])
                elif len(split_line) == 2:
                    # Word translation OOV data with labels
                    sequences_to_translate.append(split_line[0])
                elif len(split_line) == 5:
                    # OOV data as in original dataset
                    sequences_to_translate.append(split_line[0])
                else:
                    raise ValueError(
                        "Unrecognized line format {}".format(split_line))
        # Translate the list of OOV words that we just read.
        pred_translations = self.translate_list(
            sequences_to_translate, show_progbar=show_progbar, n_jobs=n_jobs,
            edit_distance=edit_distance)
        return pred_translations

    @overrides
    def translate_list(self, oov_list, show_progbar=True, n_jobs=1, debug=False,
                       edit_distance="vanilla", lang_code=None):
        if n_jobs > 1 and debug:
            raise ValueError("Cannot use multiprocessing with interactive mode.")
        # Romanize the oov_list if applicable.
        if self.uroman_path is not None:
            logger.info("Romanizing list of OOVs to translate.")
            oov_list = uromanize_list(oov_list, self.uroman_path)

        self.source_vocab = list(self.foreign_to_english.keys())
        foreign_to_english = self.foreign_to_english
        del self.foreign_to_english

        logger.info("Using edit distance type: {}".format(edit_distance))
        if n_jobs > 1:
            logger.info("Translating with {} processes".format(n_jobs))
            pool = multiprocessing.Pool(processes=n_jobs)
            # Create a multiprocess pool with the _get_nearest_source alias.
            # This is not used in python 3 because there's overhead in passing
            # the object back and forth.
            _bound_get_nearest_source_mp_alias = functools.partial(
                _get_nearest_source_mp_alias, self, edit_distance, lang_code)
            nearest_sources = pool.map(_bound_get_nearest_source_mp_alias, oov_list)
        else:
            if show_progbar:
                oov_iterable = tqdm(oov_list)
            else:
                oov_iterable = oov_list

            nearest_sources = [self._get_nearest_source(oov, debug=debug,
                                                        edit_distance=edit_distance,
                                                        lang_code=lang_code) for
                               oov in oov_iterable]
        self.foreign_to_english = foreign_to_english
        if debug:
            for nearest_source_list in nearest_sources:
                for nearest_source in nearest_source_list:
                    english_translations = self.foreign_to_english[nearest_source]
                    predicted_translations = heapq.nlargest(
                        5, english_translations.keys(),
                        key=lambda k: english_translations[k])
                    predicted_translations_with_counts = [
                        str((predicted_translation,
                             english_translations[predicted_translation])) for
                        predicted_translation in predicted_translations]
                    logger.info("Top 5 translations for source token {}: {}".format(
                        nearest_source, ", ".join(predicted_translations_with_counts)))

        predicted_translations = []
        # Iterate over the list of nearest sources produced for each OOV
        for nearest_source_list in nearest_sources:
            # List of (string, count) tuple.
            candidate_translations = []
            # Iterate over each nearest source
            for nearest_source in nearest_source_list:
                # Given a source word, get the English translations with the highest
                # weight.
                english_translations = self.foreign_to_english.get(nearest_source, None)
                if english_translations is None:
                    candidate_translations.append(("@@UNTRANSLATED_OOV@@", 0))
                else:
                    # Get the max value in the dict
                    max_count = max(list(english_translations.values()))
                    # Get all keys with this max value, and add them to
                    # candidate translations
                    candidate_translations.extend(
                        [(x, v) for x, v in english_translations.items() if
                         v == max_count])
            # Get the most probable english translation, and break ties if necessary
            if self.token_counts:
                # Get the value of the predicted translation and look for ties.
                max_value = max(candidate_translations, key=lambda k: k[1])[1]

                tie_candidates = [k for k, v in candidate_translations if
                                  v == max_value]
                # Take the max of this list with the key as the frequency
                predicted_translation = max(tie_candidates,
                                            key=lambda k: self.token_counts.get(k, 0))
            else:
                predicted_translation = max(candidate_translations, key=lambda k: k[1])[0]
            predicted_translations.append(predicted_translation)
        return predicted_translations

    def _get_nearest_source(self, oov_token, debug=False, edit_distance="vanilla",
                            lang_code=None):
        """
        Given a single OOV token, find the most similar source token(s).

        Parameters
        ----------
        oov_token: str
            The oov token to predict a translation for.
        """
        if edit_distance not in set(["vanilla", "substring", "weighted"]):
            raise ValueError("Invalid value for edit_distance {}. Expected "
                             "one of [vanilla, substring, weighted]".format(
                                 edit_distance))

        def calculate_similarity_with_input_oov(x):
            # Edit distance, higher means word are more more similar
            edit_ratio_score = 1 - (int(editdistance.eval(x, oov_token.lower())) /
                                    max(len(x), len(oov_token)))
            if edit_distance == "substring":
                longest_common_prefix_len = len(
                    os.path.commonprefix([x.lower(), oov_token.lower()]))
                longest_common_prefix_ratio = (longest_common_prefix_len /
                                               min(len(x), len(oov_token)))
                score = (0.75 * edit_ratio_score) + (0.25 * longest_common_prefix_ratio)
            else:
                score = edit_ratio_score
            return score

        source_vocab = self.source_vocab
        # Find the string in the vocab that has the highest similarity
        # with the OOV token. Only consider those that have a matching
        # prefix of at least one.
        candidate_source_tokens = [word for word in source_vocab if
                                   len(word) != 0 and len(oov_token) != 0 and
                                   (word.lower()[0] == oov_token.lower()[0])]
        if candidate_source_tokens == []:
            return "@@UNTRANSLATED_OOV@@"
        # Get the most similar source tokens, a list of string
        # If using weighted edit distance, precalculate edit distances.
        max_score = 0
        most_similar_source_tokens = []
        if edit_distance == "weighted":
            edit_distances = get_weighted_edit_distance(
                [oov_token] * len(candidate_source_tokens),
                candidate_source_tokens, lang_code1=lang_code, lang_code2=lang_code)
            # Calculate scores from the raw edit distances, and find the
            # tokens with the max score
            for idx, distance in enumerate(edit_distances):
                token = candidate_source_tokens[idx]
                score = 1 - (distance /
                             max(len(oov_token), len(token)))
                if score > max_score:
                    max_score = score
                    most_similar_source_tokens = [token]
                elif score == max_score:
                    most_similar_source_tokens.append(token)
        else:
            for token in candidate_source_tokens:
                score = calculate_similarity_with_input_oov(token)
                if score > max_score:
                    max_score = score
                    most_similar_source_tokens = [token]
                elif score == max_score:
                    most_similar_source_tokens.append(token)

        if debug:
            if debug is True:
                logger.info("Most similar source token(s): {}".format(
                    ", ".join(most_similar_source_tokens)))
        return most_similar_source_tokens
