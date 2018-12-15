import csv
from pathlib import Path
from typing import Tuple, Iterable, List

from rasa_nlu.training_data import TrainingData
from rasa_nlu.training_data.formats.markdown import MarkdownWriter
from rasa_nlu.training_data.message import Message

from src.my_types import Corpus
from src.utils import get_project_root, get_messages
from nltk.tokenize import WordPunctTokenizer


def convert_message_to_annotated_str(message: Message) -> str:
    """Convert Message object to string having annotated entities."""
    message.text = message.text.replace('(', '').replace(')', '').replace('[', '').replace(']', '')
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
    """Helper for to_tsv. Writes tuples to file."""
    with open(str(filename), 'w', encoding='utf8', newline='') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        header = ['Annotated sentence', 'Intent']
        if n_cols == 3:
            header.append('Training')
        tsv_writer.writerow(header)
        for row in tuples:
            tsv_writer.writerow([*row])


def to_tsv(corpus: Corpus, filename: Path):
    """Write corpus to tsv file specified by Path."""
    messages = get_messages(corpus)
    n = 3 if 'training' in messages[0].data else 2
    tuples = map(convert_message_line, messages)
    write_tsv(tuples, n, filename)


def annotate_entity_tokens(entity: dict) -> List[str]:
    """Create NER annotations for given entity."""
    span_generator = WordPunctTokenizer().span_tokenize(entity['value'])
    spans = [span for span in span_generator]
    entity_name = entity['entity']
    annotations = map(lambda i: ('B-' if i == 0 else 'I-') + entity_name, range(len(spans)))
    return list(annotations)


def merge_spans(spans: List[Tuple], entities: List[dict]) -> List[Tuple]:
    """Merge spans which are covered by some entity."""
    start_indexes = list(map(lambda e: e['start'], entities))
    end_indexes = list(map(lambda e: e['end'], entities))
    merged = []
    start = 0
    skip = False
    for span in spans:
        start_match = span[0] in start_indexes
        end_match = span[1] in end_indexes
        if start_match:
            start = span[0]
            skip = True
        if end_match:
            merged.append((start, span[1]))
            skip = False
        if not (start_match or end_match or skip):
            merged.append(span)
    return merged


# i need a connection from [harras](StationStart) to [karl-preis-platz](StationDest) at [8 am](TimeStartTime).
def annotate_tokens(spans: List[Tuple], entities: List[dict]) -> List[str]:
    """Use entities to create NER annotations for given spans."""
    merged_spans = merge_spans(spans, entities)
    start_indexes = list(map(lambda e: e['start'], entities))
    annotations = []
    for span in merged_spans:
        start = span[0]
        if start in start_indexes:
            entity = list(filter(lambda e: e['start'] == start, entities))[0]
            annotations.append(annotate_entity_tokens(entity))
        else:
            annotations.append('O')
    annotations = [item for sublist in annotations for item in sublist]  # flatten list
    return annotations


def convert_message_lines(message: Message) -> str:
    """Convert message to lines which can be stored in txt in the well-known NER format."""
    span_generator = WordPunctTokenizer().span_tokenize(message.text)
    spans = [span for span in span_generator]
    tokens = list(map(lambda t: message.text[t[0]:t[1]], spans))
    entities = message.data['entities'] if ('entities' in message.data) else []
    annotations = annotate_tokens(spans, entities)

    # cannot use this assertion thanks to incorrect start index for some sentence in AskUbuntuCorpus
    # Problem upgrading Ubuntu [9.10](UbuntuVersion:Ubuntu 9.10)
    # assert len(tokens) == len(annotations)
    lines = list(map(lambda t: '{} {}'.format(t[0], t[1]), zip(tokens, annotations)))
    lines = '\n'.join(lines)
    return lines


def convert_messages_lines(messages: Iterable[Message]) -> str:
    """Convert multiple messages to lines which can be stored in well-known NER format."""
    return '\n\n'.join(map(lambda message: convert_message_lines(message), messages))


def write(text: str, filename: Path):
    """Writes entire file at once based on text."""
    with open(str(filename), 'w', encoding='utf8') as f:
        f.write(text)


def write_filtered_ner(messages: Iterable[Message], filename: Path, training: bool):
    """Writes train or test messages to filename."""
    filtered_messages = filter(lambda message: message.data['training'] == training, messages)
    write(convert_messages_lines(filtered_messages), filename)


def write_ner(corpus: Corpus, folder: Path):
    """Writes complete corpus to train and test files."""
    messages = get_messages(corpus)
    write_filtered_ner(messages, folder / 'train.txt', training=True)
    write_filtered_ner(messages, folder / 'test.txt', training=False)


if __name__ == '__main__':
    dataset = Corpus.SNIPS2017
    folder = get_project_root() / 'generated' / dataset.name.lower()
    write_ner(dataset, folder)
    # to_tsv(dataset, folder / (dataset.name.lower() + '.tsv'))
