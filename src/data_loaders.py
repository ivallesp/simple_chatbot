import urllib.request
import zipfile
import io
import os
import codecs

import pandas as pd

from src.general_utilities import flatten
from src.common_paths import get_data_path


def load_cornell_dialogs(max_length = 150):
    path = os.path.join(get_data_path(), "cornell movie-dialogs corpus")
    if not os.path.exists(path):
        url = 'http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip'
        response = urllib.request.urlopen(url)
        data = response.read()  # a `bytes` object
        zip_ref = zipfile.ZipFile(io.BytesIO(data))
        zip_ref.extractall(os.path.join(get_data_path()))

    movie_lines = codecs.open(os.path.join(path, "movie_lines.txt"), "r", "Windows-1252").readlines()
    movie_lines = list(map(lambda x: x.strip().split(" +++$+++ "), movie_lines))
    movie_lines_dict = dict(list(map(lambda x:(x[0], x[-1]), movie_lines)))

    movie_conversations = open(os.path.join(path, "movie_conversations.txt"), "r").readlines()
    movie_conversations = list(map(lambda x:x.strip().split(" +++$+++ "), movie_conversations))

    for element in range(len(movie_conversations)):
        movie_conversations[element][-1] = [movie_lines_dict[line] for line in eval(movie_conversations[element][-1])]

    dialogs = flatten(list(map(lambda x:list(zip(x[-1][:-1], x[-1][1:])), movie_conversations)))
    dialogs_filtered = list(filter(lambda x: max([len(s) for s in x]) <= max_length, dialogs))
    return(dialogs_filtered)