from functools import lru_cache

from rasa_nlu.training_data.formats.markdown import MarkdownWriter
from rasa_nlu.training_data import Message, TrainingData
from typing import Iterable, Tuple
from src.corpora import read_nlu_evaluation_corpora
import pandas as pd
import json
from pathlib import Path
from src.my_types import Corpus


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def convert_json_dict(file: Path) -> dict:
    """Simple function which opens json file and returns dict."""
    with open(str(file), 'rb') as f:
        return json.load(f)


def get_path(corpus: Corpus) -> Path:
    if corpus == Corpus.MOCK:
        raise AssertionError('This function should not be called on {}.'.format(corpus))
    paths = {
        Corpus.ASKUBUNTU: Path('askubuntu') / 'original' / 'AskUbuntuCorpus.json',
        Corpus.CHATBOT: Path('chatbot') / 'original' / 'ChatbotCorpus.json',
        Corpus.WEBAPPLICATIONS: Path('webapplications') / 'original' / 'WebApplicationsCorpus.json',
        Corpus.SNIPS2017: Path('snips2017') / 'original'
    }
    return get_project_root() / 'data' / paths[corpus]


def create_entity(start: int, end: int, entity: str, value: str) -> dict:
    """Helper function to create entity which can be used in Message from rasa_nlu.training_data.message."""
    return {'start': start, 'end': end, 'entity': entity, 'value': value}


def create_message(text: str, intent: str, entities: [], training: bool, corpus: Corpus) -> Message:
    """Helper function to create a message: Message used by Rasa including whether train or test sentence."""
    message = Message.build(text, intent, entities)
    message.data['training'] = training
    message.data['corpus'] = corpus
    return message


def convert_message_to_annotated_str(message: Message) -> str:
    """Convert Message object to string having annotated entities."""
    training_examples = [message]
    training_data = TrainingData(training_examples)
    generated = MarkdownWriter()._generate_training_examples_md(training_data)
    generated = generated[generated.find('\n') + 3:-1]  # removing header
    return generated


def convert_messages_dataframe(messages: Iterable[Message], annotated_str=False) -> pd.DataFrame:
    """ Returns a DataFrame (table) from a list of Message objects which can be used for visualisation."""
    pd.set_option('max_colwidth', 180)

    data = {'message': [], 'intent': [], 'training': []}
    for message in messages:
        data['message'].append(convert_message_to_annotated_str(message) if annotated_str else message.text)
        data['intent'].append(message.data['intent'])
        data['training'].append(message.data['training'])
    return pd.DataFrame(data)


@lru_cache()
def get_messages(corpus: Corpus) -> Tuple[Message, ...]:
    """Get all messages: Message from some file containing corpus and cache the messages."""
    from src.snips import read_snips2017

    functions = {  # all functions should have type: Corpus -> Iterable[Message]
        Corpus.WEBAPPLICATIONS: read_nlu_evaluation_corpora,
        Corpus.CHATBOT: read_nlu_evaluation_corpora,
        Corpus.ASKUBUNTU: read_nlu_evaluation_corpora,
        Corpus.SNIPS2017: read_snips2017
    }
    return tuple(functions[corpus](corpus))


def get_filtered_messages(corpus: Corpus, train: bool) -> Iterable[Message]:
    return filter(lambda m: train == m.data['training'], get_messages(corpus))


def get_intents(corpus: Corpus) -> Iterable[str]:
    """ Returns intent for each message in some corpus. To get unique intents one can simply cast it to a set. """
    return map(lambda m: m.data['intent'], get_messages(corpus))
