# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import logging
import os
import six
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from oov.utils.uroman.uroman import uromanize_list

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Given language pair data, output a datafile for use in the "
                     "task of seq2seq translation of individual words (from their "
                     "characters, morphemes, or other tokenization scheme)."))
    argparser.add_argument("--data_dir", type=str, required=True,
                           help=("The path to the directory with the "
                                 "language pair data folders (e.g. hau-eng, uzb-eng). "
                                 "For each language, we will pull from the lexicon and "
                                 "tgt_given_src files."))
    argparser.add_argument("--most_frequent_only", action="store_true",
                           help=("For each source word, use only the target word "
                                 "that it is most frequently aligned to."))
    argparser.add_argument("--weight_instances", action="store_true",
                           help=("Weight each training example by the "
                                 "unnormalized t-table frequency."))
    argparser.add_argument("--romanize", action="store_true",
                           help=("Romanizes each training instance with uroman."))
    argparser.add_argument("--phrases_only", action="store_true",
                           help=("Only take instances from the train.phrases file."))
    argparser.add_argument("--output_dir", type=str, required=True,
                           help=("The path to write the generated "
                                 "data files for each language."))

    config = argparser.parse_args()
    language_pair_dirs = sorted(
        [os.path.join(config.data_dir, name) for name in
         os.listdir(config.data_dir) if
         os.path.isdir(os.path.join(config.data_dir, name))])

    for language_pair_dir in language_pair_dirs:
        translation_pairs = {}
        language_pair_str = os.path.basename(language_pair_dir)

        if not config.phrases_only:
            # Pull the data from the t-table
            t_table_file_path = os.path.join(language_pair_dir,
                                             "tgt_given_src.unnormalized")
            # Dict of dicts, each key is the source word and each value is a dictionary
            # with the associated targets in the t-table and their counts.
            source_to_target = {}
            with open(t_table_file_path) as t_table_file:
                for line in t_table_file:
                    tgt_token, src_token, count = line.split(" ")
                    if src_token not in source_to_target:
                        source_to_target[src_token] = {}
                    source_to_target[src_token][tgt_token] = float(count)

            # Add the t-table source words + their most common alignment to
            # the translation pairs
            for source_word, target_words in source_to_target.items():
                if ("@" not in source_word and "http" not in source_word and
                        source_word != "NULL" and "no_gloss" not in source_word):
                    # Filter some elements of the dictionary
                    clean_target_words = {target: target_words[target] for target
                                          in target_words if "@" not in target and
                                          "http" not in target and isEnglish(target) and
                                          target != "NULL" and
                                          target != "untranslated" and
                                          "no_gloss" not in target}
                    if len(clean_target_words) == 0:
                        continue
                    if config.most_frequent_only:
                        # Get the target word with the max probability
                        max_target_word = max(clean_target_words,
                                              key=lambda k: clean_target_words[k])
                        frequency = int(clean_target_words[max_target_word])
                        # If raw frequency is 1, skip this word
                        if frequency <= 1:
                            continue

                        # If instance weighting, count is the frequency. Else, is 1
                        if not config.weight_instances:
                            frequency = 1
                        translation_pairs[(source_word, max_target_word)] = int(frequency)
                    else:
                        # Iterate over all target words, and add them to the
                        # translation dict
                        for target_word, frequency in clean_target_words.items():
                            # If raw frequency is 1, skip this word
                            if int(frequency) <= 1:
                                continue
                            if not config.weight_instances:
                                frequency = 1
                            translation_pairs[(source_word, target_word)] = int(frequency)

        if not config.phrases_only:
            # Read the bilingual dictionary and add 1-word definitions to the
            # translation pairs dictionary with a weight of 100
            bilingual_dict_file_path = os.path.join(language_pair_dir, "lexicon")
            if not os.path.exists(bilingual_dict_file_path):
                logger.info("Bilingual dictionary for {} does "
                            "not exist at {}, not using it".format
                            (language_pair_str, bilingual_dict_file_path))
            else:
                with open(bilingual_dict_file_path) as bilingual_dict_file:
                    for line in bilingual_dict_file:
                        source_word, pos, target_word = line.rstrip("\n").split("\t")
                        if ("http" not in source_word and "@" not in source_word and
                                "http" not in target_word and "@" not in target_word and
                                isEnglish(target_word) and pos != "X" and
                                "no_gloss" not in target_word and
                                "no_gloss" not in source_word and
                                len(target_word.split()) == 1):
                            frequency = 100 if config.weight_instances else 1
                            translation_pairs[(source_word, target_word)] = max(
                                frequency, translation_pairs.get(
                                    (source_word, target_word), 0))

        # Read the phrase table and add mappings that are not yet in the dictionary
        train_phrases_path = os.path.join(language_pair_dir, "train.phrases")
        with open(train_phrases_path) as train_phrases_file:
            for line in train_phrases_file:
                source_word, target_word, frequency = line.rstrip("\n").split("\t")
                if ("http" not in source_word and "@" not in source_word and
                        "http" not in target_word and "@" not in target_word and
                        (source_word, target_word) not in translation_pairs and
                        "no_gloss" not in target_word and
                        "no_gloss" not in source_word and
                        int(frequency) > 1):
                    translation_pairs[(source_word, target_word)] = int(frequency)

        # Turn the translation pairs dictionary into a list of tuples
        translation_pairs_list = []
        for source_target_tuple, frequency in translation_pairs.items():
            source_word, target_word = source_target_tuple
            for i in range(frequency):
                translation_pairs_list.append((source_word, target_word))

        # Sort the tuples for reproducibility
        sorted_translation_pairs_list = sorted(translation_pairs_list)

        # If we are to uromanize, break the list of tuples and uromanize.
        if config.romanize:
            logger.info("Romanizing the data ({} pairs)".format(
                len(sorted_translation_pairs_list)))
            uroman_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir,
                "util", "uroman-v1.2", "bin", "uroman.pl"))

            sources, targets = zip(*sorted_translation_pairs_list)
            # Run each of the lists through uroman
            uromanized_sources = uromanize_list(sources, uroman_path)
            uromanized_targets = uromanize_list(targets, uroman_path)
            assert len(uromanized_sources) == len(uromanized_targets)
            # Recreate the list of tuples, skipping pairs with only whitespace
            sorted_translation_pairs_list = [
                (ur_source, ur_target) for
                ur_source, ur_target in zip(uromanized_sources, uromanized_targets)
                if ur_source.strip() != "" and ur_target.strip() != ""]
            logger.info("Done romanizing, kept {} pairs".format(
                len(sorted_translation_pairs_list)))

        # Write the lists of tuples to a file.
        if not os.path.exists(os.path.join(config.output_dir,
                                           language_pair_str)):
            os.makedirs(os.path.join(config.output_dir,
                                     language_pair_str))
        output_folder = os.path.join(config.output_dir, language_pair_str)

        filename = language_pair_str
        if config.phrases_only:
            filename += ".phrases_only"
        if config.most_frequent_only:
            filename += ".most_frequent"
        if config.weight_instances:
            filename += ".weighted"
        if config.romanize:
            filename += ".romanized"
        output_train_path = os.path.join(output_folder, filename +
                                         ".word_translation" + ".train")
        with open(output_train_path, 'w') as output_train_file:
            for source_word, target_word in sorted_translation_pairs_list:
                output_train_file.write("{}\t{}\n".format(
                    source_word, target_word))

        # Take the dev and test files and take only the first two columns
        # Romanize them if applicable.
        if os.path.exists(os.path.join(language_pair_dir, "dev.gold")):
            dev_file_path = os.path.join(language_pair_dir, "dev.gold")
        else:
            dev_file_path = os.path.join(language_pair_dir, "dev")
        with open(dev_file_path) as dev_file:
            dev_sources = []
            dev_targets = []
            for line in dev_file:
                source, target = line.split("\t")[:2]
                dev_sources.append(source)
                dev_targets.append(target)
        output_dev_path = os.path.join(output_folder, language_pair_str +
                                       ".word_translation.dev")
        if config.romanize:
            output_dev_path += ".romanized"
            dev_sources = uromanize_list(dev_sources, uroman_path)
            dev_targets = uromanize_list(dev_targets, uroman_path)
            assert len(dev_sources) == len(dev_targets)

        with open(output_dev_path, 'w') as output_dev_file:
            for source_word, target_word in zip(dev_sources, dev_targets):
                output_dev_file.write("{}\t{}\n".format(
                    source_word, target_word))

        # Take the test and test files and take only the first two columns
        if os.path.exists(os.path.join(language_pair_dir, "test.gold")):
            test_file_path = os.path.join(language_pair_dir, "test.gold")
        else:
            test_file_path = os.path.join(language_pair_dir, "test")
        with open(test_file_path) as test_file:
            test_sources = []
            test_targets = []
            for line in test_file:
                source, target = line.split("\t")[:2]
                test_sources.append(source)
                test_targets.append(target)
        output_test_path = os.path.join(output_folder, language_pair_str +
                                        ".word_translation.test")
        if config.romanize:
            output_test_path += ".romanized"
            test_sources = uromanize_list(test_sources, uroman_path)
            test_targets = uromanize_list(test_targets, uroman_path)
            assert len(test_sources) == len(test_targets)

        with open(output_test_path, 'w') as output_test_file:
            for source_word, target_word in zip(test_sources, test_targets):
                output_test_file.write("{}\t{}\n".format(
                    source_word, target_word))

        logger.info("Wrote {} word translation pairs to {}".format(
            language_pair_str, output_folder))


def isEnglish(s):
    if six.PY2:
        try:
            s.decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True
    if six.PY3:
        try:
            s.encode('ascii')
        except UnicodeEncodeError:
            return False
        else:
            return True


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
