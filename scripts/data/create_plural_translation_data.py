# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import ast
from difflib import SequenceMatcher
import logging
import mmap
import os
from random import shuffle
import sys

from fuzzywuzzy import fuzz
from tqdm import tqdm

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from oov.utils.uroman.uroman import uromanize_list


def main():
    argparser = argparse.ArgumentParser(
        description=("Given t-tables (in both directions), the "
                     "path to a copy of Ulf's morph cache, and a "
                     "path to uroman, output train, dev, and test "
                     "splits for the task of translating plural nouns."))
    argparser.add_argument("--tgt_given_src_path", type=str, required=True,
                           help=("The path to the weighted tgt_given_src t-table. "
                                 "The expected format is "
                                 "<english><spc><foreign><spc><weight>"))
    argparser.add_argument("--src_given_tgt_path", type=str, required=True,
                           help=("The path to the weighted src_given_tgt t-table. The "
                                 "expected format is "
                                 "<foreign><spc><english><spc><weight>"))
    argparser.add_argument("--ulf_morph_cache_path", type=str, required=True,
                           help=("The path to ulf's morph cache."))
    argparser.add_argument("--uroman_path", type=str, required=True,
                           help=("The path to uroman executable."))
    argparser.add_argument("--uroman_lang_code", type=str, required=True,
                           help=("The language code to pass to uroman."))
    argparser.add_argument("--output_folder", type=str, required=True,
                           help=("The folder to write the output files to."))
    argparser.add_argument("--output_name", type=str, required=True,
                           help=("The prefix to use when writing output."))
    argparser.add_argument("--split", action="store_true",
                           help=("Make train/test/dev splits"))

    config = argparser.parse_args()
    logger.info("Extracting singular/plural "
                "pairs from {} and {}".format(
                    config.tgt_given_src_path, config.src_given_tgt_path))
    foreign_to_english, english_to_foreign, eng_singular_plural_pairs = read_raw_data(
        config.tgt_given_src_path, config.src_given_tgt_path,
        config.ulf_morph_cache_path)
    english_foreign_plural_singular_pairs = get_singular_plural_examples(
        foreign_to_english, english_to_foreign,
        eng_singular_plural_pairs, config.uroman_path,
        config.uroman_lang_code)
    if not os.path.exists(config.output_folder):
        os.makedirs(config.output_folder)

    if not config.split:
        logger.info("Writing all singular plural pairs")
        with open(os.path.join(config.output_folder, config.output_name) +
                  ".singular-plural", "w") as outfile:
            for line in english_foreign_plural_singular_pairs:
                outfile.write("{}\t{}\t{}\t{}\n".format(
                    line[0], line[1], line[2], line[3]))
    else:
        logger.info("Generating splits and writing them to file")
        val_test_size = int(0.1 * len(english_foreign_plural_singular_pairs))
        shuffle(english_foreign_plural_singular_pairs)
        test = english_foreign_plural_singular_pairs[:val_test_size]
        val_and_train = english_foreign_plural_singular_pairs[val_test_size:]
        val = val_and_train[:val_test_size]
        train = val_and_train[val_test_size:]
        logger.info("Writing splits to file")
        for split, name in [(test, "test"), (val, "val"), (train, "train")]:
            with open(os.path.join(config.output_folder, config.output_name) +
                      ".singular-plural." + name, "w") as outfile:
                for line in split:
                    outfile.write("{}\t{}\t{}\t{}\n".format(
                        line[0], line[1], line[2], line[3]))


def get_singular_plural_examples(foreign_to_english, english_to_foreign,
                                 en_singular_plural_dict, uroman_path,
                                 uroman_lang_code):
        english_foreign_plural_singular_pairs = []
        logger.info("Searching for singular-plural foreign pairs "
                    "among {} english->foreign pairs.".format(len(english_to_foreign)))

        # Uromanize all the foreign words
        # Get a set of all the foreign words, then convert to list
        foreign_words = set()
        foreign_words.update(foreign_to_english.keys())
        for english, foreign_dict in english_to_foreign.items():
            foreign_words.update(foreign_dict.keys())
        foreign_words = list(foreign_words)
        # Pass the list of foreign words through uroman
        uromanized_foreign_words = uromanize_list(
            foreign_words, uroman_path, uroman_lang_code)
        # Build a dict from foreign words to their uromanized versions
        assert len(foreign_words) == len(uromanized_foreign_words)
        foreign_to_uroman = {
            foreign: uromanized for foreign, uromanized in
            zip(foreign_words, uromanized_foreign_words)
        }

        # Now, we look for singular-plural pairs by going through the
        # en->fr data.
        for english_singular, foreign_translations in tqdm(
                english_to_foreign.items()):
            # If the english word is not alphabetical, lowercase, or singular,
            # toss it out
            if (english_singular not in en_singular_plural_dict or not
                english_singular.islower() or not
                    english_singular.isalpha()):
                continue

            # Get the plural form of the English translation
            english_plural = en_singular_plural_dict[english_singular]
            # Throw out this pair if the English plural is the same as the singular,
            # not alphabetical, not lowercase, or not in our english->fr dictionary
            if (english_plural == english_singular or
                    not english_plural.islower() or
                    not english_plural.isalpha() or
                    english_plural not in english_to_foreign):
                continue

            # Get a list of foreign words that translate to the English singular
            foreign_singular_translations = [
                foreign_singular for foreign_singular in foreign_translations.keys() if
                foreign_singular != english_singular and
                foreign_singular != english_plural and isclean(foreign_singular)]
            # Get the 5 foreign translations of the english word with the highest weight.
            best_foreign_singular_translations = sorted(
                foreign_singular_translations, key=lambda x: foreign_translations[x],
                reverse=True)[:5]
            if best_foreign_singular_translations == []:
                continue

            # Get all the 5 foreign (hopefully plural) words that aligned most frequently
            # to the English plural.
            best_foreign_plural_translations = sorted(
                [foreign_plural for foreign_plural in
                 english_to_foreign[english_plural] if isclean(foreign_plural) and
                 foreign_plural != english_singular and foreign_plural != english_plural],
                key=lambda foreign: english_to_foreign[english_plural][foreign],
                reverse=True)[:5]
            if best_foreign_plural_translations == []:
                continue

            # At this point, we have 5 candidate singular foreign words and
            # 5 candidate plural foreign words. We want to find the pair
            # that is most likely to be a singular-plural pair.
            foreign_singular_plural_scores = []
            for singular_translation in best_foreign_singular_translations:
                for plural_translation in best_foreign_plural_translations:
                    uromanized_singular_translation = foreign_to_uroman[
                        singular_translation]
                    uromanized_plural_translation = foreign_to_uroman[
                        plural_translation]

                    if uromanized_singular_translation == uromanized_plural_translation:
                        continue
                    if (len(uromanized_singular_translation) >
                            len(uromanized_plural_translation)):
                        continue

                    # Set some score cutoffs
                    fuzz_ratio = fuzz.ratio(uromanized_singular_translation,
                                            uromanized_plural_translation)
                    if fuzz_ratio < 50:
                        continue
                    fuzz_partial_ratio = fuzz.partial_ratio(
                        uromanized_singular_translation,
                        uromanized_plural_translation)
                    if fuzz_partial_ratio < 50:
                        continue
                    pair_substring_similarity_ratio = substring_similarity_ratio(
                        uromanized_singular_translation, uromanized_plural_translation)
                    if pair_substring_similarity_ratio < 0.50:
                        continue

                    # score the pair and add them to list of scores
                    pair_score = (fuzz_ratio + fuzz_partial_ratio +
                                  (pair_substring_similarity_ratio * 150))
                    foreign_singular_plural_scores.append(
                        (singular_translation,
                         plural_translation, pair_score))
            if foreign_singular_plural_scores == []:
                continue
            else:
                max_score = max(foreign_singular_plural_scores, key=lambda x: x[2])[2]
                best_foreign_pairs = [
                    (sing, pl) for sing, pl, score in
                    foreign_singular_plural_scores if score == max_score]
                if len(best_foreign_pairs) == 1:
                    best_foreign_singular, best_foreign_plural = best_foreign_pairs[0]
                else:
                    # Break ties by running the score function on non-uromanized output
                    best_foreign_singular, best_foreign_plural = max(
                        best_foreign_pairs,
                        key=lambda x: (fuzz.ratio(x[0], x[1]) +
                                       fuzz.partial_ratio(x[0], x[1]) +
                                       substring_similarity_ratio(x[0], x[1])))
            english_foreign_plural_singular_pairs.append(
                (english_singular, best_foreign_singular,
                 english_plural, best_foreign_plural))

        if len(english_foreign_plural_singular_pairs) == 0:
            raise ValueError("Didn't find any singular/plural pairs in the data!")
        logger.info("Found {} english-foreign singular-plural pairs".format(len(
            english_foreign_plural_singular_pairs)))
        return english_foreign_plural_singular_pairs


def read_raw_data(tgt_given_src_path, src_given_tgt_path, ulf_morph_cache_path):
    # Read the tgt_given_src alignments and generate a dict of
    # foreign word to counter, where the keys of the counter are english translations
    # and the value of the counter is how many times the alignment occurred.
    foreign_to_english = {}
    logger.info("Reading foreign to english "
                "alignments from {}".format(tgt_given_src_path))
    with open(tgt_given_src_path) as foreign_to_english_file:
        for line in tqdm(foreign_to_english_file,
                         total=get_line_number(tgt_given_src_path)):
            english, foreign, weight = line.rstrip("\n").split(" ")
            weight = float(weight)
            if foreign not in foreign_to_english:
                foreign_to_english[foreign] = {}
            foreign_to_english[foreign][english] = weight

    # Read the src_given_tgt alignments and generate a dict of English word
    # to Counter, where the keys of the Counter are English translations and
    # the value of the counter is how many times the alignment occurred.
    english_to_foreign = {}
    logger.info("Reading english to foreign "
                "alignments from {}".format(src_given_tgt_path))
    with open(src_given_tgt_path) as english_to_foreign_file:
        for line in tqdm(english_to_foreign_file,
                         total=get_line_number(src_given_tgt_path)):
            foreign, english, weight = line.rstrip("\n").split(" ")
            weight = float(weight)
            if english not in english_to_foreign:
                english_to_foreign[english] = {}
            english_to_foreign[english][foreign] = weight

    # Read the Ulf morph cache and generate a dict of English singular->plural pairs
    logger.info("Reading the Ulf morph cache from {}".format(ulf_morph_cache_path))
    eng_singular_plural_pairs = {}
    removed_words = set()
    with open(ulf_morph_cache_path) as ulf_morph_cache_file:
        for line in tqdm(ulf_morph_cache_file,
                         total=get_line_number(ulf_morph_cache_path)):
            # skip comments
            if "#" == line[0]:
                continue

            # Extract the surface form and singular form
            field_split = line.rstrip("\n").split(" ::")
            plural_form = ast.literal_eval(field_split[0].lstrip("::SURF "))
            singular_form = ast.literal_eval(field_split[1].lstrip("LEX "))
            pos = field_split[2].lstrip("SYNT ")

            # If either word is already in the dictionary
            # remove it and add it to the skipped set and skip this one
            if (singular_form in eng_singular_plural_pairs.keys() or
                singular_form in eng_singular_plural_pairs.values() or
                plural_form in eng_singular_plural_pairs.keys() or
                    plural_form in eng_singular_plural_pairs.values()):

                if "NOUN" not in pos:
                    eng_singular_plural_pairs.pop(singular_form, None)
                    eng_singular_plural_pairs.pop(plural_form, None)
                    eng_singular_plural_pairs = {
                        k: v for k, v in
                        eng_singular_plural_pairs.items() if
                        v != plural_form or v != singular_form}
                    removed_words.add(singular_form)
                    removed_words.add(plural_form)
                continue

            # Skip the pairs if it is in the removed pairs
            if singular_form in removed_words or plural_form in removed_words:
                continue

            # Verify the words have a number feature and are nouns
            if ":NUMBER" not in line or "NOUN" not in line:
                continue

            # Extract the number and verify it is plural
            number_field = [field for field in field_split if ":NUMBER" in
                            field]
            if len(number_field) != 1:
                continue
            number = number_field[0][(number_field[0].find(":NUMBER ") +
                                      len(":NUMBER")):
                                     number_field[0].rfind(" :")].strip()
            if number != "F-PLURAL":
                continue

            eng_singular_plural_pairs[singular_form] = plural_form
    return (foreign_to_english, english_to_foreign, eng_singular_plural_pairs)


def substring_similarity_ratio(str1, str2):
    match = SequenceMatcher(None, str1, str2).find_longest_match(
        0, len(str1), 0, len(str2))
    return (match.size * 2) / (len(str1) + len(str2))


def isclean(input_string):
    if any(char.isdigit() for char in input_string):
        return False
    if any(char == "," or char == "." or char == "[" or
           char == "]" or char == "-" for char in input_string):
        return False
    return True


def get_line_number(file_path):
    """
    Utility to get the number of lines in a file.
    """
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
