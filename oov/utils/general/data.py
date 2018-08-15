import mmap


def get_num_lines(file_path):
    """
    Given a string path to a file, return the number of
    lines in it.
    """
    with open(file_path, "r+") as file_handle:
        buf = mmap.mmap(file_handle.fileno(), 0)
        lines = 0
        while buf.readline():
            lines += 1
        return lines
