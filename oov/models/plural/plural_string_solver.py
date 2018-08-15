from __future__ import unicode_literals

import torch

from collections import Counter
from difflib import SequenceMatcher
import logging
import os
import pprint

import dill
import inflect
import numpy as np
from overrides import overrides
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from ..base_solver import BaseSolver
from ...utils.general.data import get_num_lines

logger = logging.getLogger(__name__)


class PluralStringSolver(BaseSolver):
    """
    Broadly, this class translates plural strings where the singular version is
    in our vocabulary (we know its English translation). When training the model,
    it learns both the string transformations associated with singularization, as
    well as where they apply. The task is cast as one of classification --- given
    features associated with a plural string, predict the proper string transformation
    to apply in order to singularize.

    When we have the foreign singular, we can translate the foreign singular into an
    English singular. We can then pluralize this English singular to form an English
    plural, and return that as the predicted translation.

    At test time, we featurize each OOV in the same fashion and then
    predict a string transformation to singularize it. We then translate that
    singular foreign word to English, and pluralize the English word and use
    it as our predicted translation.
    """

    def __init__(self):
        self.solver_init_params = locals()
        self.solver_init_params.pop("self")
        self.solver_init_params.pop("__class__", None)

        self.foreign_to_english = None
        self.label_to_idx = None
        self.idx_to_label = None
        self.ngrams = None
        self.singularization_transforms = None
        self.clf = None
        self.was_loaded = False

    @overrides
    def get_state_dict(self):
        state_dict = {
            "solver_class": self.__class__,
            "solver_init_params": self.solver_init_params,
            "ngrams": self.ngrams,
            "singularization_transforms": self.singularization_transforms,
            "idx_to_label": self.idx_to_label,
            "label_to_idx": self.label_to_idx,
            "clf": self.clf,
            "foreign_to_english": self.foreign_to_english
        }
        return state_dict

    @overrides
    def load_from_state_dict(self, state_dict):
        if not set(["foreign_to_english", "ngrams",
                    "singularization_transforms", "idx_to_label", "label_to_idx",
                    "clf"]).issubset(set(state_dict.keys())):
            raise ValueError("state dict had unexpected "
                             "keys {}".format(set(state_dict.keys())))
        self.foreign_to_english = state_dict["foreign_to_english"]
        self.ngrams = state_dict["ngrams"]
        self.singularization_transforms = state_dict["singularization_transforms"]
        self.label_to_idx = state_dict["label_to_idx"]
        self.idx_to_label = state_dict["idx_to_label"]
        self.clf = state_dict["clf"]
        self.was_loaded = True
        return self

    @overrides
    def load_from_file(self, load_path):
        state_dict = torch.load(load_path)
        return self.load_from_state_dict(state_dict)

    @overrides
    def read_data(self, english_foreign_singular_plural_train, tgt_given_src_path,
                  english_foreign_singular_plural_val=None, en_wordlist_path=None):
        """
        Read the data necessary to train this solver. In particular, we need
        a file of (english singular, foreign singular, english plural,
        foreign plural) pairs a collection of (pretrained, if
        possible on a large corpus) FastText vectors.

        Parameters
        ----------
        english_foreign_singular_plural_train: str
            Path to file where each line is:
            <english singular>\t<foreign singular>\t<english plural>\t<foreign plural>
            giving the pluralization of the english singular, as well as translations
            of both of them.

        tgt_given_src_path: str
            The path to the weighted tgt_given_src t-table. The expected format is
            <english><spc><foreign><spc><weight>. We use this to form a bilingual
            dictionary of foreign word to English translation.

        english_foreign_singular_plural_val: str, optional (default=None)
            Path to file where each line is:
            <english singular>\t<foreign singular>\t<english plural>\t<foreign plural>
            giving the pluralization of the english singular, as well as translations
            of both of them. Model accuracy will be evaluated with this data after
            training.

        en_wordlist_path: str, optional (default=None)
            Path to a wordlist (one word per line). Translations will be constrained
            to be within this wordlist (the wordlist will be our English vocabulary).

        Returns
        -------
        data_dict: dict
            data_dict is a dictionary, where the various keys are meant to be used
            as input to the kwargs of the train_model function.
            The \"train_foreign_singular_plural_pairs\" key maps to a list of tuples
            used to train the solver, where each tuple has
            (foreign singular, foreign plural). Similarily, the
            \"val_foreign_singular_plural_pairs\" maps to a list of tuples used in
            validation of the solver, where each tuple is again of (foreign singular,
            foreign plural).
        """
        if self.was_loaded:
            raise ValueError("PluralStringSolver does not support "
                             "training from a saved model.")
        train_foreign_singular_plural_pairs = []
        # Read the english foreign singular plural data to generate
        # a list of foreign singular/plurals
        logger.info("Reading training english foreign singular "
                    "plural data from {}".format(english_foreign_singular_plural_train))
        with open(english_foreign_singular_plural_train) as en_fr_sing_plural_train:
            for line in tqdm(
                    en_fr_sing_plural_train,
                    total=get_num_lines(english_foreign_singular_plural_train)):
                _, fr_singular, _, fr_plural = line.rstrip("\n").split("\t")
                train_foreign_singular_plural_pairs.append((fr_singular, fr_plural))

        # Read the english foreign singular plural data to generate a list of
        # (english singular, foreign singular, english plural, foreign plural) pairs.
        logger.info("Reading validation english foreign singular "
                    "plural data from {}".format(english_foreign_singular_plural_val))
        if english_foreign_singular_plural_val:
            val_english_foreign_singular_plural_pairs = []
            with open(english_foreign_singular_plural_val) as en_fr_sing_plural_val_data:
                for line in tqdm(
                        en_fr_sing_plural_val_data,
                        total=get_num_lines(english_foreign_singular_plural_val)):
                    en_singular, fr_singular, en_plural, fr_plural = line.rstrip(
                        "\n").split("\t")
                    val_english_foreign_singular_plural_pairs.append(
                        (en_singular, fr_singular, en_plural, fr_plural))
        else:
            val_english_foreign_singular_plural_pairs = None

        # Read the tgt_given_src alignments and generate a dict of
        # foreign word to most likely English translation
        foreign_to_english_counter = {}
        logger.info("Reading foreign to english "
                    "alignments from {}".format(tgt_given_src_path))
        with open(tgt_given_src_path) as foreign_to_english_file:
            num_skipped = 0
            for line in tqdm(foreign_to_english_file,
                             total=get_num_lines(tgt_given_src_path)):
                split_line = line.rstrip("\n").split(" ")
                if len(split_line) == 3:
                    english, foreign, weight = split_line
                else:
                    num_skipped += 1
                    continue
                if foreign not in foreign_to_english_counter:
                    foreign_to_english_counter[foreign] = {}
                foreign_to_english_counter[foreign][english] = float(weight)
            if num_skipped != 0:
                logger.warning("Skipped {} lines in tgt_given_src due to improper "
                               "whitespace format.".format(num_skipped))

        # Read the wordlist and use it to filter the foreign_to_english dictionary
        english_vocabulary = None
        if en_wordlist_path:
            with open(en_wordlist_path) as en_wordlist_file:
                english_vocabulary = [line.rstrip("\n") for line in en_wordlist_file]
                english_vocabulary = set(english_vocabulary)

        foreign_to_english = {}
        for foreign_word, english_translations in foreign_to_english_counter.items():
            if english_vocabulary:
                # Get a list of english translations that are in our vocabulary
                # and their weights.
                candidate_english_translations = [
                    (x, x_weight) for x, x_weight in
                    english_translations.items() if x in english_vocabulary]
            else:
                candidate_english_translations = [
                    (x, x_weight) for x, x_weight in english_translations.items()]
            if candidate_english_translations == []:
                continue
            foreign_to_english[foreign_word] = max(candidate_english_translations,
                                                   key=lambda x: x[1])[0]

        self.foreign_to_english = foreign_to_english
        return {"train_foreign_singular_plural_pairs":
                train_foreign_singular_plural_pairs,
                "val_english_foreign_singular_plural_pairs":
                val_english_foreign_singular_plural_pairs}

    @overrides
    def save_to_file(self, save_dir, run_id):
        save_path = os.path.join(save_dir, run_id + "_model.pkl")
        state_dict = self.get_state_dict()
        torch.save(state_dict, save_path, pickle_module=dill)

    @overrides
    def train_model(self, train_foreign_singular_plural_pairs,
                    val_english_foreign_singular_plural_pairs=None,
                    plural_freq_thres=1, log_dir=None, save_dir=None,
                    run_id=None):
        """
        Train a model by extracting both the string transformations used to
        singularize a word, as well as the common environments that distinguish
        them.

        Parameters
        ----------
        train_foreign_singular_plural_pairs,: List of tuple
            train_foreign_singular_plural_pairs is a list of tuples, where each
            tuple is (foreign singular, foreign plural).

        val_english_foreign_singular_plural_pairs: List of tuple, optional (default=None)
            val_english_foreign_singular_plural_pairs is a list of tuples, where
            each tuple is (english singular, foreign singular, english plural,
            foreign plural)

        plural_freq_thres: int, optional (default=None)
            The frequency threshold for a plural transformation in order to use
            it as a label for training. Any plural string transformation observed
            less than plural_freq_thres will be discarded.

        Returns
        -------
        clf:
            Returns the classifier fit on the data. The model is also saved to self.clf
        """
        if self.was_loaded:
            raise ValueError("PluralStringSolver does not support "
                             "training from a saved model.")
        # TODO: If a log file is provided, write train progress to it.
        logger.info("Training model")
        raw_str_labels, singularization_transforms = self._get_singularization_transforms(
            train_foreign_singular_plural_pairs, plural_freq_thres)

        # singularization_transforms is already frequency filtered
        self.singularization_transforms = singularization_transforms
        logger.info("Filtering training instances to include only those "
                    "whose the label is above freq_thres.")
        if len(raw_str_labels) != len(train_foreign_singular_plural_pairs):
            raise ValueError("Generated a {} labels, but have {} train pairs".format(
                len(raw_str_labels), len(train_foreign_singular_plural_pairs)))
        # Remove train instances if the label is not in deplurization_transforms
        train_foreign_singular_plural_pairs = [
            x for idx, x in enumerate(train_foreign_singular_plural_pairs)
            if raw_str_labels[idx] in singularization_transforms]
        str_labels = [x for x in raw_str_labels if x in
                      singularization_transforms]

        # Convert the labels to numbers
        self.label_to_idx = {}
        for label in singularization_transforms:
            self.label_to_idx[label] = len(self.label_to_idx)
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        encoded_labels = [self.label_to_idx[label] for label in str_labels]

        # Feature Types: (1-3) character bigrams, trigrams, and 4-grams
        # (4) Whether applying the transformation would produce an in-vocab word.
        ngrams = self._get_ngrams(
            train_foreign_singular_plural_pairs)
        self.ngrams = ngrams

        # Turn list of foreign plurals into a 2D numpy array
        # of shape (num_foreign_plurals, num_ngrams + (num_singularization_transforms))
        foreign_plural_feature_vectors = self._featurize(
            list(zip(*train_foreign_singular_plural_pairs))[1])
        logger.info("Training model")
        self.clf = RandomForestClassifier(n_estimators=300)
        self.clf = self.clf.fit(foreign_plural_feature_vectors, encoded_labels)
        if val_english_foreign_singular_plural_pairs:
            val_foreign_singular = list(zip(
                *val_english_foreign_singular_plural_pairs))[1]
            val_english_plural = list(zip(*val_english_foreign_singular_plural_pairs))[2]
            val_foreign_plural = list(zip(*val_english_foreign_singular_plural_pairs))[3]
            predicted_singulars, predicted_translations = self._translate_list(
                val_foreign_plural)
            # Compare val_foreign_singular and predicted_singular, and
            # print singularization accuray
            num_correct_singularizations = 0
            for pred_singular, gold_singular, gold_plural in zip(predicted_singulars,
                                                                 val_foreign_singular,
                                                                 val_foreign_plural):
                if pred_singular.lower() == gold_singular.lower():
                    num_correct_singularizations += 1
            logger.info("Singularization Validation Accuracy: {}".format(
                num_correct_singularizations / len(val_foreign_singular)))

            # Compare val_english_plural and predicted_translations, and
            # print translation accuracy
            num_correct_translations = 0
            for pred_translation, gold_translation in zip(predicted_translations,
                                                          val_english_plural):
                if pred_translation.lower() == gold_translation.lower():
                    num_correct_translations += 1
            logger.info("Translation Validation Accuracy: {}".format(
                num_correct_translations / len(val_english_plural)))
        if save_dir is not None and run_id is not None:
            logger.info("Saving trained model to save dir {} with run "
                        "id {}".format(save_dir, run_id))
            self.save_to_file(save_dir=save_dir, run_id=run_id)
        return self.clf

    @overrides
    def translate_file(self, oov_path, show_progbar=True, n_jobs=1):
        """
        Parameters
        ----------
        oov_path: str
            Path to file with OOV data. The PluralStringSolver takes
            three kinds of data --- either OOV data from the original dataset,
            OOV data as generated by the create_word_translation_data script,
            the specialized plural translation data generated by the
            create_plural_translation_data script, or a file containing
            just foreign OOVs to translate.

        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar. Unused in this class.
        """
        sequences_to_translate = []
        num_cols = None
        with open(oov_path) as oov_file:
            for line in oov_file:
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
                elif len(split_line) == 4:
                    # Plural translation data
                    sequences_to_translate.append(split_line[3])
                elif len(split_line) == 5:
                    # OOV data as in original dataset
                    sequences_to_translate.append(split_line[0])
                else:
                    raise ValueError(
                        "Unrecognized line format {}".format(split_line))
        # Translate the list of OOV words that we just read.
        pred_translations = self.translate_list(
            sequences_to_translate, show_progbar=show_progbar, n_jobs=n_jobs)
        return pred_translations

    @overrides
    def translate_list(self, oov_list, show_progbar=True, n_jobs=1, debug=False):
        """
        show_progbar: boolean, optional (default=True)
            Whether or not to show a progress bar. Unused in this class.
        """
        _, pred_translations = self._translate_list(oov_list)
        return pred_translations

    def _apply_transforms(self, string, transforms):
        """
        Given a string and a singularization transform, apply the transform
        to the string.

        Parameters
        ----------
        string: str
            The string to apply the transform to.

        transforms: frozenset
            A frozenset of string, dictating the string transformations
            to apply in order to singularize the input string.
        """
        string = "^" + string + "$"
        # First delete the things that should be deleted
        for transform in transforms:
            if transform.startswith("-"):
                # Apply this transform by deleting this part of the string
                string = string.replace(transform[1:], "")
        # Now add on the things that should be added
        for transform in transforms:
            if transform.startswith("+"):
                transform = transform[1:]
                # Prepend if it starts with ^, else append
                if transform.startswith("^"):
                    string = transform + string
                else:
                    string = string + transform
        string = string.lstrip("^").rstrip("$")
        return string

    def _featurize(self, foreign_plurals):
        """
        Given a list of foreign plurals, use the set of ngrams and plural
        affixes that we previously read to featurize a list of foreign plural words.

        Parameters
        ----------
        foreign_plurals: List of str
            List of strings, where each string is a foreign plural word.

        Returns
        -------
        feature_matrix: ndarray
            feature_matrix is of shape
            (num_foreign_plurals, num_ngrams + (num_singularization_transforms))
            and represents the featurized input foreign plural words.
        """
        logger.info("Featurizing input, each input "
                    "is vector of size {}".format(
                        len(self.ngrams) + (2 * len(self.singularization_transforms))))
        # Preallocate the feature matrix, it's of size:
        # (num_foreign_plurals) X (num_ngrams + num_singularization_transforms)
        feature_matrix_shape = (len(foreign_plurals),
                                len(self.ngrams) + len(self.singularization_transforms))

        feature_matrix = np.zeros(feature_matrix_shape)
        for plural_idx, foreign_plural in tqdm(enumerate(foreign_plurals),
                                               total=len(foreign_plurals)):
            foreign_plural = "^" + foreign_plural + "$"
            feature_idx = 0
            # Fill in the feature matrix for ngrams
            for ngram in self.ngrams:
                if ngram in foreign_plural:
                    feature_matrix[plural_idx][feature_idx] = 1
                feature_idx += 1
            # Fill in the feature matrix for if applying an affix generates an
            # in-vocabulary word.
            for singularization_transform in self.singularization_transforms:
                # Check whether this this singularization transform can apply.
                can_apply = True
                for plural_affix in singularization_transform:
                    if (plural_affix.startswith("-") and not
                            plural_affix[1:] in foreign_plural):
                        can_apply = False
                        continue
                if can_apply:
                    # Fill in the feature matrix for if the transform results in a
                    # valid foreign vocab word.
                    if (self._apply_transforms(foreign_plural,
                                               singularization_transform)
                            in self.foreign_to_english):
                        feature_matrix[plural_idx][feature_idx] = 1
                feature_idx += 1
        assert feature_idx == (len(self.ngrams) + len(self.singularization_transforms))
        return feature_matrix

    def _get_singularization_transforms(self, foreign_singular_plural_pairs,
                                        freq_thres=1):
        """
        Parameters
        ----------
        foreign_singular_plural_pairs: List of tuple
            foreign_singular_plural_pairs is a list of tuples, where each
            tuple is (foreign singular, foreign plural).

        freq_thres: int, optional (default=1)
            The minimum number of occurences a singularization transformation
            must be observed for it to be included in the returned dictionary.

        Returns
        -------
        singularization_tuple: tuple of (List, set)
            singularization_tuple is a tuple of (List, set). The first List holds
            the appropriate singularization transformation for input foreign/singular
            pair. The second element, a set, is a set of frozensets where each frozenset
            represents an individual singularization transform.
        """
        all_singularization_transforms = Counter()
        sample_singularization_transforms = []
        for fr_singular, fr_plural in foreign_singular_plural_pairs:
            singular_plural_matcher = SequenceMatcher(None, fr_singular, fr_plural)
            match = singular_plural_matcher.find_longest_match(0, len(fr_singular),
                                                               0, len(fr_plural))
            longest_common_substring = fr_singular[match.a: match.a + match.size]
            if longest_common_substring == "":
                logger.warning("singular {},  plural {} has no common "
                               "substring".format(fr_singular, fr_plural))
                longest_common_substring = " "
            # Now, with this longest common substring, we want to slice off strings
            # from the ends of the plural until we get it.
            fr_plural = "^" + fr_plural + "$"
            to_subtract = set(fr_plural.split(longest_common_substring))
            to_subtract.difference_update(["", "^", "$"])
            to_subtract = frozenset(map(lambda x: "-" + x, to_subtract))

            # Now we want to add things to the longest common substring until we get
            # the singular version.
            fr_singular = "^" + fr_singular + "$"
            to_add = set(fr_singular.split(longest_common_substring))
            to_add.difference_update(["", "^", "$"])
            to_add = frozenset(map(lambda x: "+" + x, to_add))

            # Combine the sets of segments to add and subtract
            to_add_or_subtract = to_add.union(to_subtract)
            sample_singularization_transforms.append(to_add_or_subtract)
            all_singularization_transforms[to_add_or_subtract] += 1

        total_observed = sum(all_singularization_transforms.values())
        logger.info("Total singularization transforms "
                    "observed: {}".format(total_observed))
        logger.info("Found {} unique singularization transforms, pruning those "
                    "that occur less than freq_thres ({})".format(
                        len(all_singularization_transforms), freq_thres))
        all_singularization_transforms_set = {
            x for x, count in all_singularization_transforms.items() if
            count >= freq_thres}
        logger.info("Found {} unique singularization transforms after pruning".format(
            len(all_singularization_transforms_set)))

        # Print the most common singularization transforms and the proportions
        logger.info("5 most common singularization transforms and their proportions")
        pp = pprint.PrettyPrinter(indent=4)
        for transform, count in all_singularization_transforms.most_common(5):
            pp.pprint((transform, count / total_observed))

        return sample_singularization_transforms, all_singularization_transforms_set

    def _get_ngrams(self, foreign_singular_plural_pairs):
        """
        Given foreign singular and plural pairs, extract ngrams that occur in the plural
        foreign words. In addition, we are provided a list of singularization transforms
        and must extract those that involve the plural (so subtracting affixes and such).

        Parameters
        ----------
        foreign_singular_plural_pairs: List of str tuple
            A List of tuples of strings, where each tuple is
            (foreign_singular, foreign_plural).

        Returns
        -------
        all_ngrams: set of String
            Set of strings with all the ngrams extracted from the words.
        """
        # Get a set of bigrams, trigrams, 4grams in the data
        all_ngrams = set()
        for _, fr_plural in foreign_singular_plural_pairs:
            fr_plural = "^" + fr_plural + "$"
            all_ngrams.update(set(map(''.join, self._get_word_ngrams(fr_plural, 2))))
            all_ngrams.update(set(map(''.join, self._get_word_ngrams(fr_plural, 3))))
            all_ngrams.update(set(map(''.join, self._get_word_ngrams(fr_plural, 4))))
        return all_ngrams

    def _get_word_ngrams(self, input, ngram_size):
        """
        Given an iterable input, return ngrams of size ngram_size.

        Parameters
        ----------
        input: Iterable
            Iterable (e.g. List or string) to split into ngrams.

        ngram_size: int
            The desired size of each ngram.

        Returns
        -------
        ngram_set: Set of tuples of string
            A set of tuples, where each tuple contains one character of the
            ngram.
        """
        return set(zip(*[input[i:] for i in range(ngram_size)]))

    def _translate_list(self, oov_list):
        """
        Given a list of OOV words, return predicted singularizations and
        translations for each of them.

        Parameters
        ----------
        oov_list: List of str
            List of OOV strings to translate.

        Returns
        -------
        predicted_singularizations_list: List of str
            List of predicted singularizations (foreign plural to foreign singular).

        predicted_translation_list: List of str
            List of predicted translations.
        """
        if not (self.clf and self.label_to_idx and self.idx_to_label and
                self.singularization_transforms and self.ngrams):
            raise ValueError("Must call train_model before making predictions")
        # Turn list of foreign plurals into a 2D numpy array
        # of shape (num_foreign_plurals, num_ngrams + (2*num_singularization_transforms))
        foreign_plural_feature_vectors = self._featurize(oov_list)
        logger.info("Generating translations for {} words".format(len(oov_list)))
        predicted_int_transform_classes = self.clf.predict(
            foreign_plural_feature_vectors)
        # Convert the predicted numerical indices to string transformation labels
        predicted_transform_classes = [self.idx_to_label[idx] for idx in
                                       predicted_int_transform_classes]

        # Apply the predicted transforms
        predicted_singulars = []
        for oov, predicted_transforms in zip(oov_list, predicted_transform_classes):
            predicted_singulars.append(
                self._apply_transforms(oov, predicted_transforms))

        # Check if the predicted singulars are in our bilingual dictionary,
        # and translate them to English and pluralize if they are.
        num_singular_oov = 0
        predicted_translations = []
        inflector = inflect.engine()
        for predicted_singular in predicted_singulars:
            if predicted_singular not in self.foreign_to_english:
                predicted_translations.append("OOV-singular-form-{}".format(
                    predicted_singular))
                num_singular_oov += 1
                continue
            predicted_translations.append(
                inflector.plural_noun(self.foreign_to_english[predicted_singular]))
        logger.info("{} words were not in t-table (either because they were "
                    "incorrectly singularized or simply were nt in the data), "
                    "and thus could not be translated".format(num_singular_oov))
        return predicted_singulars, predicted_translations
