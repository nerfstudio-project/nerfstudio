import random
import string
from datetime import datetime


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
