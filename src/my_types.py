from enum import Enum, auto


class Corpus(Enum):
    """Defining corpora using this enum allows for more readable code down the line."""
    MOCK = auto()  # used for testing
    ASKUBUNTU = auto()
    CHATBOT = auto()
    WEBAPPLICATIONS = auto()
    SNIPS2017 = auto()
