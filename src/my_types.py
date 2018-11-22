from enum import Enum, auto
from typing import NamedTuple, Tuple


class Corpus(Enum):
    """Defining corpora using this enum allows for more readable code down the line."""
    MOCK = auto()  # used for testing
    ASKUBUNTU = auto()
    CHATBOT = auto()
    WEBAPPLICATIONS = auto()
    SNIPS2017 = auto()


Xy = NamedTuple('Xy', [('X', Tuple), ('y', Tuple)])

DoubleSplit = NamedTuple('DoubleSplit', [('train', Xy), ('test', Xy)])

TripleSplit = NamedTuple('TripleSplit', [('train', Xy), ('dev', Xy), ('test', Xy)])
