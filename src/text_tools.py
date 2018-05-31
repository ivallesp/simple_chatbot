import nltk
from src.general_utilities import flatten
from nltk import ngrams
import numpy as np
from collections import Counter


def remove_substrings(s, substrings):
    for substr in substrings:
        s=s.replace(substr, "")
    return s

def pad(x, max_length, symbol=None, mode="left"):
    if len(x) < max_length:
        elements_to_add = max_length - len(x)
        if mode=="left":
            x = [symbol]*elements_to_add + x
        if mode=="right":
            x = x + [symbol]*elements_to_add
    return x