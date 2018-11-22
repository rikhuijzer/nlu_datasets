import src.utils  # avoiding ImportError
import functools
from src.my_types import Corpus
from rasa_nlu.training_data.message import Message


def test_message_converters():
    text = 'Could I pay you 50 yen tomorrow or tomorrow?'
    text = 'Could I pay you 50 [yen](currency lorem ipsum) [tomorrow](date) or [tomorrow](date)?'
    entities = [
        src.utils.create_entity(19, 22, 'currency lorem ipsum', 'yen'),
        src.utils.create_entity(23, 31, 'date', 'tomorrow'),
        src.utils.create_entity(35, 43, 'date', 'tomorrow')
    ]
    message = Message.build(text, 'foo', entities)

    assert text, src.utils.convert_message_str(message)
    assert message, src.utils.convert_str_message(text)


def test_get_intents():
    expected = {'Delete Account', 'Find Alternative', 'Download Video',
                'Filter Spam', 'Change Password', 'Sync Accounts', 'None', 'Export Data'}
    assert expected == set(src.utils.get_intents(Corpus.WEBAPPLICATIONS))


def test_get_filtered_messages():
    func = functools.partial(src.utils.get_filtered_messages, corpus=Corpus.MOCK)
    assert 15 == len(tuple(func(train=True)))
    assert 5 == len(tuple(func(train=False)))
