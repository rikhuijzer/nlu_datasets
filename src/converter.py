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
    for span in spans:
        if span[0] in start_indexes:
            start = span[0]
        elif span[1] in end_indexes:
            merged.append((start, span[1]))
        else:
            merged.append(span)
    return merged


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
    annotations = annotate_tokens(spans, message.data['entities'])
    lines = list(map(lambda t: '{} {}'.format(t[0], t[1]), zip(tokens, annotations)))
    lines = '\n'.join(lines)
    return lines


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
