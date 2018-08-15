import logging
import os
import six
import subprocess
import tempfile

logger = logging.getLogger(__name__)


def uromanize_list(input_list, uroman_path, lang_code=None):
    """
    Given a list of strings to uromanize, write them to a
    TemporaryFile, uromanize them, and then parse the output.

    Parameters
    ----------
    input_list: List of str
        List of string to uromanize

    uroman_path: str
        Path to the uroman perl executable.

    lang_code: str
        Optional 3-letter language code to pass to uroman.

    Returns
    -------
    uromanized_input: List of str
        Same list that was passed as input, except the contents are romanized.
    """
    # Write the input list to a tmpfile
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tempfile_path = tmp.name
    with open(tempfile_path, "w") as tmp_file:
        for input in input_list:
            if six.PY2:
                input = input.encode("utf-8")
            tmp_file.write(input + "\n")

    # Build the command to run
    if lang_code is not None:
        cmd = uroman_path + " -l " + lang_code + " < " + tempfile_path
    else:
        cmd = uroman_path + " < " + tempfile_path
    logger.info("Running command {} to uromanize".format(cmd))
    uroman_output = subprocess.check_output(cmd, shell=True,
                                            universal_newlines=True)

    # Delete the tempfile
    os.unlink(tempfile_path)
    # Parse the uroman output
    uromanized_input = [word for word in
                        uroman_output.rstrip("\n").split("\n")]
    assert len(uromanized_input) == len(input_list)
    return uromanized_input
