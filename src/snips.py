import operator
import pathlib
import typing
from itertools import accumulate, chain
from typing import List, Iterable
from src.my_types import Corpus
import src.utils  # from ... import ... will cause circular imports
from rasa_nlu.training_data.message import Message


def get_folders(corpus: Corpus) -> Iterable[pathlib.Path]:
    """Get all folders listed in some corpus folder."""
    return filter(lambda f: f.is_dir(), src.utils.get_path(corpus).glob('./*'))


def convert_data_text(data: List[dict]) -> str:
    return ''.join([item['text'] for item in data])


def convert_data_spans(data: List[dict]) -> Iterable[typing.Tuple[int, int]]:
    """Get start and end index for part of a sentence, see SNIPS 2017 .json files for examples."""
    lengths = list(map(lambda item: len(item['text']), data))
    start_indexes = [0] + list(accumulate(lengths, operator.add))[:-1]
    end_indexes = map(lambda x: x[0] + x[1], zip(start_indexes, lengths))
    return zip(start_indexes, end_indexes)


def convert_data_entities(data: List[dict]) -> Iterable[dict]:
    """Returns entities in Rasa representation for some SNIPS data element."""
    spans = convert_data_spans(data)
    for span, item in zip(spans, data):
        if 'entity' in item:
            yield src.utils.create_entity(start=span[0], end=span[1], entity=item['entity'], value=item['text'])


def convert_data_message(corpus: Corpus, intent: str, data: List[dict], train: bool) -> Message:
    """Returns message in Rasa representation for some SNIPS data element."""
    text = convert_data_text(data)
    entities = list(convert_data_entities(data))
    return src.utils.create_message(text, intent, entities, train, corpus)


def convert_file_messages(corpus: Corpus, file: pathlib.Path, intent: str, train: bool) -> Iterable[Message]:
    """Returns messages in Rasa representation for some SNIPS .json file."""
    js = src.utils.convert_json_dict(file)
    return map(lambda item: convert_data_message(corpus, intent, item['data'], train), js[intent])


def read_snips2017(corpus: Corpus) -> Iterable[Message]:
    def get_train_test_messages(folder: pathlib.Path, train: bool) -> Iterable[Message]:
        intent = folder.name
        filename = 'train_{}.json' if train else 'validate_{}.json'
        path = folder / filename.format(intent)
        return convert_file_messages(corpus, path, intent, train=train)

    def get_messages(folder: pathlib.Path) -> Iterable[Message]:
        return chain(get_train_test_messages(folder, train=True), get_train_test_messages(folder, train=False))

    folders = get_folders(corpus)
    nested_messages = map(lambda folder: get_messages(folder), folders)
    return chain.from_iterable(nested_messages)
