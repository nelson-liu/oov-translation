from __future__ import division, print_function
import argparse
import logging
import os
from tabulate import tabulate

logger = logging.getLogger(__name__)


def main():
    argparser = argparse.ArgumentParser(
        description=("Score predictions from OOV translators."))
    argparser.add_argument("--data_dir", type=str, required=True,
                           help=("The path to the directory with the "
                                 "language pair data folders (e.g. hau-eng, uzb-eng). "
                                 "The output should be written in each of the language "
                                 "pair folders as either dev.guess or test.guess."))
    argparser.add_argument("--run_id", type=str, required=True, nargs="+",
                           help=("The identifying run id that you wish to score. If "
                                 "you pass in multiple, an answer will be marked "
                                 "correct if it is correct in the predictions of "
                                 "any of the specified runs."))
    argparser.add_argument("--use_test", action="store_true",
                           help=("Evaluate on the test labels as well."))
    argparser.add_argument("--use_gold", action="store_true",
                           help=("Evaluate on the gold labels as well."))
    argparser.add_argument("--match_one", action="store_true",
                           help=("If set, take the guess file matching the run_id "
                                 "with the shortest name as the guess file to "
                                 "evaluate"))
    argparser.add_argument("--pretty", type=str, nargs='?', default=False,
                           choices=["simple", "grid", "latex"],
                           help=("Pretty print the table in a format."))
    argparser.add_argument("--overall_only", action="store_true",
                           help=("Do not output the scores for individual "
                                 "language pairs, only the dataset-wide "
                                 "score."))

    config = argparser.parse_args()
    language_pair_dirs = sorted(
        [os.path.join(config.data_dir, name) for name in
         os.listdir(config.data_dir) if
         os.path.isdir(os.path.join(config.data_dir, name)) and "-" in name])

    scores = []
    for language_pair_dir in language_pair_dirs:
        language_pair_str = os.path.basename(language_pair_dir)
        language_pair_scores = score_language_pair(
            language_pair_dir, config.run_id, config.use_test,
            config.use_gold, config.match_one)
        if language_pair_scores:
            language_pair_scores.insert(0, language_pair_str)
            scores.append(language_pair_scores)

    # Calculate overall averages across all language pairs.
    # Zip the unpacked 2D list, but ignore the first column (since those are strings).
    # Then, for each zipped item (a column), averages it by first converting all
    # the elements to floats, summing, and then dividing by the num_language_pairs
    dataset_averages = []
    for column in zip(*[row[1:] for row in scores]):
        column_scores = [float(cell) for cell in column if cell is not None]
        column_average = truncate(sum(column_scores) / len(column_scores), 3)
        dataset_averages.append(column_average)

    dataset_averages.insert(0, "Overall Accuracy")
    # Remove the language pair scores if overall_only is set
    if config.overall_only:
        scores = [dataset_averages]
    else:
        scores.append(dataset_averages)

    # Add header to table
    header = ["", "silver dev"]
    if config.use_test:
        header.append("silver test")

    if config.use_gold:
        header.append("gold dev")
        if config.use_test:
            header.append("gold test")

    scores.insert(0, header)

    if config.pretty is not False:
        tablefmt = config.pretty if config.pretty else "grid"
        print(tabulate(scores, headers="firstrow", tablefmt=tablefmt))
    else:
        print(simple_format_table(scores))


def score_language_pair(language_pair_dir, run_ids, use_test, use_gold,
                        match_one):
    """
    Score the dev, test, and whatever other evaluation data there is
    for a given language pair.

    Parameters
    ----------
    language_pair_dir: str
        The directory corresponding to a language pair.

    run_ids: List of str
        The identifying strings for the runs of the model to score.

    use_test: boolean
        Whether or not to use the test data, vs.
        just the dev data.

    use_gold: boolean
        Whether or not to use the gold-labeled data, vs.
        the silver-labeled data.

    match_one: boolean
        Whether or not to only score the shortest file that matches
        the run_id.
    """
    # If there's no guess directory, just skip
    if not os.path.exists(os.path.join(language_pair_dir, "guess")):
        return

    label_filenames = ["dev"]
    if use_test:
        label_filenames.append("test")
    if use_gold:
        label_filenames.append("dev.gold")
        if use_test:
            label_filenames.append("test.gold")

    label_paths = list(map(
        lambda x: os.path.join(language_pair_dir, x),
        label_filenames))

    # Score a pair of files
    language_pair_scores = []
    for label_path in label_paths:
        guess_files = sorted(
            [os.path.join(language_pair_dir, "guess", name) for
             name in os.listdir(os.path.join(language_pair_dir, "guess"))
             if os.path.basename(label_path) in name])
        guess_paths = []
        for run_id in run_ids:
            run_id_guess_paths = []
            for guess_file in guess_files:
                if all([id_segment in guess_file for id_segment
                        in run_id.split("_")]):
                    run_id_guess_paths.append(guess_file)
            if match_one and run_id_guess_paths != []:
                run_id_guess_paths = [min(run_id_guess_paths, key=len)]
            guess_paths.extend(run_id_guess_paths)

        # If there is no guess path, just skip
        if guess_paths == []:
            language_pair_scores.append(None)
            continue

        logger.info("Using guess files {}".format(guess_paths))

        # If gold doesn't exist, don't evaluate on it.
        if not os.path.exists(label_path):
            if ".gold" in os.path.basename(label_path):
                language_pair_scores.append(None)
                continue
            else:
                raise ValueError("Missing silver "
                                 "labels file {}".format(label_path))

        # Open the guess files
        guess_files = [open(guess_path) for guess_path in guess_paths]

        with open(label_path) as label_file:
            num_instances = 0
            num_correct = 0
            for combined_label_guess_line in zip(label_file, *guess_files):
                # Split the label from the guesses
                label_line = combined_label_guess_line[:1]
                guess_lines = combined_label_guess_line[1:]
                assert len(label_line) == 1

                guesses = set([guess_line.rstrip("\n") for guess_line in
                               guess_lines])
                num_instances += 1
                label = label_line[0].rstrip("\n").split("\t")[1]
                if label in guesses:
                    num_correct += 1
            language_pair_scores.append(truncate(num_correct / num_instances, 3))

        # Close all the guess files
        for fd in guess_files:
            fd.close()
    return language_pair_scores


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
