from __future__ import print_function
import argparse
import logging
import os

from tabulate import tabulate
from tqdm import tqdm

logger = logging.getLogger(__name__)

STATISTICS_TO_GATHER = ["multiword", "discontiguous"]
LABEL_FILENAMES = ["dev", "dev.gold"]


def main():
    argparser = argparse.ArgumentParser(
        description=("Go through the OOV dataset "
                     "(dev, test, dev.gold, test.gold files) "
                     "and print simple count-based statistics."))
    argparser.add_argument("--data_dir", type=str, required=True,
                           help=("The path to the directory with the "
                                 "language pair data files to load."))
    argparser.add_argument("--pretty", type=str, nargs='?', default=False,
                           choices=["simple", "grid", "latex", "orgtbl"],
                           help=("Pretty print the table in a format."))
    config = argparser.parse_args()

    language_data_dirs = sorted(
        [os.path.join(config.data_dir, name) for name in
         os.listdir(config.data_dir) if
         os.path.isdir(os.path.join(config.data_dir, name))])

    for statistic in STATISTICS_TO_GATHER:
        logger.info("Gathering {} statistics for {}".format(statistic,
                                                            LABEL_FILENAMES))
        all_language_data_statistic = []
        for language_data_dir in tqdm(language_data_dirs):
            language_data_str = os.path.basename(language_data_dir)
            language_data_statistics = get_language_data_statistic(
                language_data_dir, mode=statistic)
            language_data_statistics.insert(0, language_data_str)
            all_language_data_statistic.append(language_data_statistics)
        # Add header to table
        header = [""]
        for label_filename in LABEL_FILENAMES:
            header.append(label_filename + " " + statistic + " proportion")
        all_language_data_statistic.insert(0, header)
        if config.pretty is not False:
            tablefmt = config.pretty if config.pretty else "grid"
            print(tabulate(all_language_data_statistic, headers="firstrow",
                           tablefmt=tablefmt))
        else:
            print(simple_format_table(all_language_data_statistic))


def get_language_data_statistic(language_data_dir, mode):
    """
    Gather statistics for the dev data of one language pair.

    Parameters
    ----------
    language_data_dir: str
        The directory corresponding to the language data to analyze.

    mode: str
        A string describing the statistic to calculate.
    """

    label_paths = list(map(
        lambda x: os.path.join(language_data_dir, x),
        LABEL_FILENAMES))

    # Analyze the language data files
    language_data_stats = []
    for label_path in label_paths:
        # If gold doesn't exist, don't read it
        if not os.path.exists(label_path):
            if ".gold" in os.path.basename(label_path):
                language_data_stats.append(None)
                continue

        with open(label_path) as label_file:
            num_instances = 0
            num_statistic = 0
            for label_line in label_file:
                num_instances += 1
                label_src, label_tgt = label_line.rstrip("\n").split("\t")[:2]
                # Count multiword targets
                if mode == "multiword":
                    if len(label_tgt.split(" ")) > 1:
                        num_statistic += 1
                # Count discontiguous targets
                if mode == "discontiguous":
                    if "..." in label_tgt:
                        num_statistic += 1
            language_data_stats.append(truncate(num_statistic / num_instances, 3))
    return language_data_stats


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
