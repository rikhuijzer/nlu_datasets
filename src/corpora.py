from typing import Iterable, List

from rasa_nlu.training_data.message import Message

import src.utils  # got import error when using from ... import ...
from nltk.tokenize import WordPunctTokenizer
from src.my_types import Corpus


def convert_index(text: str, token_index: int, start: bool) -> int:
    """ Convert token_index as used by NLU-Evaluation Corpora to character index. """
    span_generator = WordPunctTokenizer().span_tokenize(text)
    spans = [span for span in span_generator]
    return spans[token_index][0 if start else 1]


def convert_nlu_evaluation_entity(text: str, entity: dict) -> dict:
    """ Convert a NLU Evaluation Corpora sentence to Entity object. See test for examples. """
    start = convert_index(text, entity['start'], start=True)
    end = convert_index(text, entity['stop'], start=False)
    return src.utils.create_entity(start, end, entity=entity['entity'], value=entity['text'])


def read_nlu_evaluation_corpora(corpus: Corpus) -> Iterable[Message]:
    """Convert NLU Evaluation Corpora dictionary to the internal representation."""
    file = src.utils.get_path(corpus)
    js = src.utils.convert_json_dict(file)

    def convert_entities(sentence: dict) -> List[dict]:
        return list(map(lambda e: convert_nlu_evaluation_entity(sentence['text'], e), sentence['entities']))

    def convert_sentence(sentence: dict) -> Message:
        return Message.build(text=sentence['text'], intent=sentence['intent'], entities=convert_entities(sentence))

    return map(convert_sentence, js['sentences'])
