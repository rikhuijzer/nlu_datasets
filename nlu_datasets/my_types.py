from enum import Enum, auto


class Corpus(Enum):
    """Defining corpora using this enum allows for more readable code down the line."""
    MOCK = auto()  # used for testing
    ASKUBUNTU = auto()
    CHATBOT = auto()
    WEBAPPLICATIONS = auto()
    SNIPS2017 = auto()


class Task(Enum):
    NER = auto()  # not adding any intent information to data
    INTENT = auto()  # not adding any ner information to data
    NER_INTENT = auto()  # both ner and intent information is added to data
