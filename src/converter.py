import csv
from pathlib import Path
from typing import Tuple, Iterable

from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data.formats.markdown import MarkdownWriter
from rasa_nlu.training_data.message import Message

from src.my_types import Corpus
from src.utils import get_project_root, get_messages


def convert_message_to_annotated_str(message: Message) -> str:
    """Convert Message object to string having annotated entities."""
    # message.text = message.text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
    training_examples = [message]
    training_data = TrainingData(training_examples=training_examples)
    generated = MarkdownWriter()._generate_training_examples_md(training_data)
    generated = generated[generated.find('\n') + 3:-1]  # removes header
    return generated


def convert_message_line(message: Message) -> Tuple:
    """Convert message to tuple which can be stored in tsv."""
    text = convert_message_to_annotated_str(message)
    base_row = (text, message.data['intent'])
    if 'training' in message.data:
        return (*base_row, message.data['training'])
    else:
        return base_row


def write_tsv(tuples: Iterable[Tuple], n_cols: int, filename: Path):
    with open(str(filename), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        header = ['Annotated sentence', 'Intent']
        if n_cols == 3:
            header.append('Training')
        tsv_writer.writerow(header)
        for row in tuples:
            tsv_writer.writerow([*row])


def to_tsv(corpus: Corpus, filename: Path):
    messages = get_messages(corpus)
    n = 3 if 'training' in messages[0].data else 2
    tuples = map(convert_message_line, messages)
    write_tsv(tuples, n, filename)


if __name__ == '__main__':
    dataset = Corpus.SNIPS2017
    to_tsv(dataset, get_project_root() / 'data' / dataset.name.lower() / (dataset.name.lower() + '.tsv'))
