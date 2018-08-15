import logging
import os
import six
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def get_weighted_edit_distance(list1, list2, weighted_edit_distance_path=None,
                               lang_code1=None, lang_code2=None):
    """
    Given a list of strings to uromanize, write them to a
    TemporaryFile, uromanize them, and then parse the output.

    Parameters
    ----------
    list1: List of str
        List of strings to compare to list2.

    list2: List of str
        List of strings to compare to list1.

    weighted_edit_distance_path: str or None
        Path to the weighted edit distance script.

    lang_code1: str
        Optional 3-letter language code to pass
        representing the language of list1.

    lang_code2: str
        Optional 3-letter language code to pass
        representing the language of list2.

    Returns
    -------
    edit_distances: List of float
    """
    if len(list1) != len(list2):
        raise ValueError("Lengths of two lists to get edit "
                         "distance of are not the same.")
    # Write the input lists to a tmpfile
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tempfile_path = tmp.name
    with open(tempfile_path, "w") as tmp_file:
        for input1, input2 in zip(list1, list2):
            if six.PY2:
                input1 = input1.encode("utf-8")
                input2 = input2.encode("utf-8")
            tmp_file.write("{}\t{}\n".format(input1, input2))

    if weighted_edit_distance_path is None:
        weighted_edit_distance_path = os.path.abspath(os.path.realpath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir,
            "util", "weighted-ed-v0.2", "bin", "weighted-ed.pl")))
    # Build the command to run
    cmd = weighted_edit_distance_path
    if lang_code1 is not None:
        cmd += " -lc1 " + lang_code1
    if lang_code2 is not None:
        cmd += " -lc2 " + lang_code2
    cmd += " < " + tempfile_path
    logger.info("Running command {} to get edit distances".format(cmd))
    weighted_ed_output = subprocess.check_output(cmd, shell=True,
                                                 universal_newlines=True)

    # Delete the tempfile
    os.unlink(tempfile_path)
    # Parse the output
    output_edit_distances = [float(ed) for ed in
                             weighted_ed_output.rstrip("\n").split("\n")]
    assert len(output_edit_distances) == len(list1)
    return output_edit_distances
