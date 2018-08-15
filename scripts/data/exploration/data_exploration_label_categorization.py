from __future__ import division, print_function
import argparse
from collections import Counter
import logging
import os
import random
import subprocess
import tempfile

from Levenshtein import distance
from tabulate import tabulate
from tqdm import tqdm

logger = logging.getLogger(__name__)

LABEL_FILENAMES = ["dev", "dev.gold"]


def main():
    argparser = argparse.ArgumentParser(
        description=("Go through the OOV dataset "
                     "(dev, test, dev.gold, test.gold files) "
                     "and print simple count-based statistics."))
    argparser.add_argument("--data_dir", type=str, required=True,
                           help=("The path to the directory with the "
                                 "language pair data files to load."))
    argparser.add_argument("--uroman_path", type=str, required=True,
                           help=("The path to the uroman executable."))
    argparser.add_argument("--pretty", type=str, nargs='?', default=False,
                           choices=["simple", "grid", "latex", "orgtbl"],
                           help=("Pretty print the table in a format."))
    config = argparser.parse_args()

    language_data_dirs = sorted(
        [os.path.join(config.data_dir, name) for name in
         os.listdir(config.data_dir) if
         os.path.isdir(os.path.join(config.data_dir, name))])

    logger.info("Calculating the proportion of examples involving affixation.")
    all_affix_ratios = []
    for language_data_dir in tqdm(language_data_dirs):
        language_data_str = os.path.basename(language_data_dir)
        affix_ratios, affix_translations = calculate_affixation_proportion(
            language_data_dir)
        affix_ratios.insert(0, language_data_str)
        all_affix_ratios.append(affix_ratios)
    header = ["Language Pair"]
    for label_filename in LABEL_FILENAMES:
        header.append(label_filename + " affixation")
    all_affix_ratios.insert(0, header)
    if config.pretty is not False:
        tablefmt = config.pretty if config.pretty else "grid"
        print(tabulate(all_affix_ratios, headers="firstrow",
                       tablefmt=tablefmt))
    else:
        print(simple_format_table(all_affix_ratios))

    logger.info("Calculating the proportion of examples involving compounding.")
    all_compound_ratios = []
    for language_data_dir in tqdm(language_data_dirs):
        language_data_str = os.path.basename(language_data_dir)
        compound_ratios, compound_translations = calculate_compound_proportion(
            language_data_dir)
        compound_ratios.insert(0, language_data_str)
        all_compound_ratios.append(compound_ratios)
    header = ["Language Pair"]
    for label_filename in LABEL_FILENAMES:
        header.append(label_filename + " compounds")
    all_compound_ratios.insert(0, header)
    if config.pretty is not False:
        tablefmt = config.pretty if config.pretty else "grid"
        print(tabulate(all_compound_ratios, headers="firstrow",
                       tablefmt=tablefmt))
    else:
        print(simple_format_table(all_compound_ratios))

    logger.info("Calculating the proportion of examples involving "
                "misspellings (edit distance <= 2)")
    all_misspelling_ratios = []
    for language_data_dir in tqdm(language_data_dirs):
        language_data_str = os.path.basename(language_data_dir)
        misspelling_ratios = calculate_misspelling_proportion(
            language_data_dir, config.uroman_path)
        misspelling_ratios.insert(0, language_data_str)
        all_misspelling_ratios.append(misspelling_ratios)
    header = ["Language Pair"]
    for label_filename in LABEL_FILENAMES:
        header.append(label_filename + " close edit distance")
    all_misspelling_ratios.insert(0, header)
    if config.pretty is not False:
        tablefmt = config.pretty if config.pretty else "grid"
        print(tabulate(all_misspelling_ratios, headers="firstrow",
                       tablefmt=tablefmt))
    else:
        print(simple_format_table(all_misspelling_ratios))


def calculate_affixation_proportion(language_data_dir):
    """
    Calculate the amount of dev set examples that involve affixation.


    Parameters
    ----------
    language_data_dir: str
        The directory corresponding to the language data to analyze.
    """

    label_paths = list(map(
        lambda x: os.path.join(language_data_dir, x),
        LABEL_FILENAMES))

    # Put the t-table into a dictionary
    src_to_target = {}
    tgt_given_src_unnorm_path = os.path.join(language_data_dir,
                                             "tgt_given_src.unnormalized")
    with open(tgt_given_src_unnorm_path, "r") as tgt_given_src_unnorm_file:
        for line in tgt_given_src_unnorm_file:
            tgt_token, src_token, count = line.split(" ")
            if src_token not in src_to_target:
                src_to_target[src_token] = {}
            src_to_target[src_token][tgt_token] = float(count)

    # Loop through the source words and get frequencies of each affix:
    affix_counts = Counter()
    for source_word in src_to_target.keys():
        # affixes are all the ways to split a string
        for i in range(1, len(source_word)):
            split_one, split_two = source_word[:i], source_word[i:]
            affix_counts["-" + split_one] += 1
            affix_counts[split_two + "-"] += 1

    results = []
    # Analyze the language data files
    for label_path in label_paths:
        # If gold doesn't exist, don't read it
        if not os.path.exists(label_path):
            if ".gold" in os.path.basename(label_path):
                results.append(None)
                continue

        with open(label_path) as label_file:
            affix_translations = {}
            num_instances = 0
            for label_line in label_file:
                num_instances += 1
                label_src, label_tgt = label_line.rstrip("\n").split("\t")[:2]

                # get all possible splits of label_src
                for i in range(1, len(label_src)):
                    source_split_one, source_split_two = label_src[:i], label_src[i:]
                    # Get all splits for the target
                    for j in range(1, len(label_tgt)):
                        target_split_one, target_split_two = label_tgt[:j], label_tgt[j:]
                        # Check if either of the source splits translate
                        # to either of the target splits.
                        source_split_one_translations = src_to_target.get(
                            source_split_one, {})
                        source_split_two_translations = src_to_target.get(
                            source_split_two, {})
                        translated = [(source_split_one, source_split_two),
                                      (target_split_one, target_split_two)]
                        if (source_split_one_translations.get(target_split_one, 0) > 1 and
                                affix_counts[source_split_two + "-"] > 500):
                            affix_translations[label_src] = translated + [(1, 1)]
                            continue
                        if (source_split_one_translations.get(target_split_two, 0) > 1 and
                                affix_counts[source_split_two + "-"] > 500):
                            affix_translations[label_src] = translated + [(1, 2)]
                            continue
                        if (source_split_two_translations.get(target_split_one, 0) > 1 and
                                affix_counts["-" + source_split_one] > 500):
                            affix_translations[label_src] = translated + [(2, 1)]
                            continue
                        if (source_split_two_translations.get(target_split_two, 0) > 1 and
                                affix_counts["-" + source_split_one] > 500):
                            affix_translations[label_src] = translated + [(2, 2)]
                            continue
        results.append(truncate(len(affix_translations) / num_instances, 2))
    # Get 5 random samples from the gold dev set if applicable or dev set if not
    random_affix_translations_subset = {k: v for k, v in
                                        random.sample(affix_translations.items(), 5)}
    return results, random_affix_translations_subset


def calculate_compound_proportion(language_data_dir):
    """
    Calculate the number of dev set examples that involve compounding.

    Parameters
    ----------
    language_data_dir: str
        The directory corresponding to the language data to analyze.
    """

    label_paths = list(map(
        lambda x: os.path.join(language_data_dir, x),
        LABEL_FILENAMES))

    # Put the t-table into a dictionary
    src_to_target = {}
    tgt_given_src_unnorm_path = os.path.join(language_data_dir,
                                             "tgt_given_src.unnormalized")
    with open(tgt_given_src_unnorm_path, "r") as tgt_given_src_unnorm_file:
        for line in tgt_given_src_unnorm_file:
            tgt_token, src_token, count = line.split(" ")
            if src_token not in src_to_target:
                src_to_target[src_token] = {}
            src_to_target[src_token][tgt_token] = float(count)

    results = []
    # Analyze the language data files
    for label_path in label_paths:
        # If gold doesn't exist, don't read it
        if not os.path.exists(label_path):
            if ".gold" in os.path.basename(label_path):
                results.append(None)
                continue

        with open(label_path) as label_file:
            compound_translations = {}
            num_instances = 0
            for label_line in label_file:
                num_instances += 1
                label_src, label_tgt = label_line.rstrip("\n").split("\t")[:2]

                # get all possible splits of label_src
                for i in range(1, len(label_src)):
                    source_split_one, source_split_two = label_src[:i], label_src[i:]
                    # Get all splits for the target
                    for j in range(1, len(label_tgt)):
                        target_split_one, target_split_two = label_tgt[:j], label_tgt[j:]
                        # Check if there is a 1-1 translation correspondence
                        source_split_one_translations = src_to_target.get(
                            source_split_one, {})
                        source_split_two_translations = src_to_target.get(
                            source_split_two, {})
                        translated = [(source_split_one, source_split_two),
                                      (target_split_one, target_split_two)]
                        if (source_split_one_translations.get(
                                target_split_one, 0) > 1 and
                                source_split_two_translations.get(
                                    target_split_two, 0) > 1):
                            compound_translations[label_src] = translated + [(1, 1)]
                            continue
                        if (source_split_one_translations.get(
                                target_split_two, 0) > 1 and
                                source_split_two_translations.get(
                                    target_split_one, 0) > 1):
                            compound_translations[label_src] = translated + [(1, 2)]
                            continue

        results.append(truncate(len(compound_translations) / num_instances, 2))
    # Get 5 random samples from the gold dev set if applicable or dev set if not
    random_compound_translations_subset = {
        k: v for k, v in random.sample(compound_translations.items(),
                                       min(5, len(compound_translations)))}
    return results, random_compound_translations_subset


def calculate_misspelling_proportion(language_data_dir, uroman_path):
    """
    Calculate the amount of dev set examples that have small edit distance
    (less than or equal to 2) and are thus likely misspellings.

    Parameters
    ----------
    language_data_dir: str
        The directory corresponding to the language data to analyze.

    uroman_path: str
        The path to the uroman executable.
    """

    label_paths = list(map(
        lambda x: os.path.join(language_data_dir, x),
        LABEL_FILENAMES))

    results = []
    # Analyze the language data files
    for label_path in label_paths:
        # If gold doesn't exist, don't read it
        if not os.path.exists(label_path):
            if ".gold" in os.path.basename(label_path):
                results.append(None)
                continue

        label_srcs = []
        label_targets = []
        with open(label_path) as label_file:
            num_instances = 0
            num_misspellings = 0
            for label_line in label_file:
                num_instances += 1
                label_src, label_tgt = label_line.rstrip("\n").split("\t")[:2]
                label_srcs.append(label_src)
                label_targets.append(label_tgt)
            # Write the label srcs to a file to be uromanized
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tempfile_path = tmp.name
            with open(tempfile_path, "w") as tmp_file:
                for label_src in label_srcs:
                    tmp_file.write(label_src + "\n")

        cmd = uroman_path + " < " + tempfile_path
        logger.info("Running command {} to uromanize".format(cmd))
        uroman_output = subprocess.check_output(cmd, shell=True,
                                                universal_newlines=True)
        os.unlink(tempfile_path)

        # Parse the uroman output
        uromanized_label_srcs = [word for word in
                                 uroman_output.rstrip("\n").split("\n")]

        for idx, uromanized_src in enumerate(uromanized_label_srcs):
            label_tgt = label_targets[idx]
            if distance(uromanized_src, label_tgt) <= 2:
                num_misspellings += 1
        results.append(truncate(num_misspellings / num_instances, 2))
    return results


def truncate(input_float, num_decimals):
    """
    Truncates/pads a float to num_decimals decimal places without rounding.

    Parameters
    ----------
    input_float: float
        The float to truncate.

    num_decimals: int
        The number of decimal places to preserve.
    """
    s = '{}'.format(input_float)
    if 'e' in s or 'E' in s:
        return '{0:.{1}f}'.format(input_float, num_decimals)
    i, p, d = s.partition('.')
    return '.'.join([i, (d + '0' * num_decimals)[:num_decimals]])


def simple_format_table(table):
    """
    Given a 2D list as a table, format it with simple python functions.
    This exists because we ideally would like to not have to depend on
    tabulate or other pip packages in order to evaluate.

    Parameters
    ----------
    table: List of Lists
        A List of Lists representing a table with variable columns
        and rows to be pretty printed.
    """
    s = [[str(e) for e in row] for row in table]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    return '\n'.join(table)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                        "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
