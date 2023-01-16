from pathlib import Path
import random
import string
from datetime import datetime
import sys


def get_timestamp():
    return datetime.today().strftime("%Y-%m-%d-%H-%M-%S")


def random_codeword():
    """ac53"""
    letters = random.sample(string.ascii_lowercase, 2)
    word = "".join(letters)
    return f"{word}_{random.randint(10, 99)}"


def get_experiment_name(timestamp=None, codeword=None):
    timestamp = timestamp if timestamp else get_timestamp()
    codeword = codeword if codeword else random_codeword()
    return f"{timestamp}-{codeword}"


class SocketConcatenator(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
        self.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def stdout_to_file(file: Path):
    """
    Pipes standard input to standard input and to a file.
    """
    print("Standard output and errors piped to file:")
    f = open(Path(file), "w")
    sys.stdout = SocketConcatenator(sys.stdout, f)
    sys.stderr = SocketConcatenator(sys.stderr, f)


def reset_sockets():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
