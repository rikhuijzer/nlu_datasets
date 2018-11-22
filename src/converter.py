import csv
from pathlib import Path
from typing import Tuple, Optional, List, Iterable

from rasa_nlu.training_data.message import Message
from sklearn.model_selection import train_test_split

from src.utils import get_project_root, get_messages
from src.my_types import Corpus, Xy, DoubleSplit, TripleSplit


def convert_message_line(message: Message, train) -> Optional[Tuple]:
    if message.data['training'] == train:
        return message.text, message.data['intent']


def write_tsv(xys: List[Xy], filename: Path):
    with open(str(filename), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        for X, y in zip(xys.X, xys.y):
            tsv_writer.writerow([X, y])


def get_x_y(messages: Tuple[Message, ...]) -> Xy:
    x = list(map(lambda message: message.text, messages))
    y = list(map(lambda message: message.data['intent'], messages))
    return Xy(x, y)


def my_train_test_split(x, y, split_size: float) -> Tuple:
    """Returns train_test_split using random state 0 and stratified on y."""
    return train_test_split(x, y, test_size=split_size, random_state=0, stratify=y)


def get_double_split(x: List, y: List) -> DoubleSplit:
    """Returns stratified train / test split (75 / 25)."""
    x_train, x_test, y_train, y_test = my_train_test_split(x, y, split_size=0.25)
    train = Xy(x_train, y_train)
    test = Xy(x_test, y_test)
    return DoubleSplit(train, test)


def get_triple_split(x: List, y: List) -> TripleSplit:
    """Return stratified train / dev / test split (60 / 20 / 20)."""
    x_train, x_dev, y_train, y_dev = my_train_test_split(x, y, split_size=0.20)
    x_train, x_test, y_train, y_test = my_train_test_split(x_train, y_train, split_size=0.25)

    train = Xy(x_train, y_train)
    dev = Xy(x_dev, y_dev)
    test = Xy(x_test, y_test)
    return TripleSplit(train, dev, test)


def to_tsv(corpus: Corpus, folder: Path):
    messages = get_messages(corpus)
    split = get_triple_split(*get_x_y(messages))
    write_tsv(split.train, folder / 'train.tsv')
    write_tsv(split.dev, folder / 'dev.tsv')
    write_tsv(split.test, folder / 'test.tsv')


if __name__ == '__main__':
    to_tsv(Corpus.ASKUBUNTU, get_project_root() / 'generated' / 'askubuntu')
