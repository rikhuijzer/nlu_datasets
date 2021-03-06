import nlu_datasets.utils  # avoiding ImportError
import functools
from nlu_datasets.my_types import Corpus
from rasa_nlu.training_data.message import Message


def test_message_converters():
    text = 'Could I pay you 50 yen tomorrow or tomorrow?'
    entities = [
        nlu_datasets.utils.create_entity(19, 22, 'currency lorem ipsum', 'yen'),
        nlu_datasets.utils.create_entity(23, 31, 'date', 'tomorrow'),
        nlu_datasets.utils.create_entity(35, 43, 'date', 'tomorrow')
    ]
    message = Message.build(text, 'foo', entities)

    expected = 'Could I pay you 50 [yen](currency lorem ipsum) [tomorrow](date) or [tomorrow](date)?'
    assert expected == nlu_datasets.utils.convert_message_annotated(message)


def test_get_intents():
    expected = {'Delete Account', 'Find Alternative', 'Download Video',
                'Filter Spam', 'Change Password', 'Sync Accounts', 'None', 'Export Data'}
    assert expected == set(nlu_datasets.utils.get_intents(Corpus.WEBAPPLICATIONS))


def test_get_filtered_messages():
    func = functools.partial(nlu_datasets.utils.get_filtered_messages, corpus=Corpus.MOCK)
    assert 15 == len(tuple(func(train=True)))
    assert 5 == len(tuple(func(train=False)))
