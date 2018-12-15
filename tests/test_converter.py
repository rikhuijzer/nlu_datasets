from src.utils import create_message
from src.my_types import Corpus
from src.converter import (
    convert_message_lines, annotate_tokens, annotate_entity_tokens, merge_spans, convert_messages_lines
)

message = create_message(text='Alternative to Facebook Messenger.',
                         intent='Find Alternative',
                         entities=[{'end': 33,
                                    'entity': 'WebService',
                                    'start': 15,
                                    'value': 'Facebook Messenger'}],
                         training=False,
                         corpus=Corpus.MOCK)
entities = message.data['entities']
spans = [(0, 11), (12, 14), (15, 23), (24, 33), (33, 34)]
lines = [
    'Alternative O',
    'to O',
    'Facebook B-WebService',
    'Messenger I-WebService',
    '. O'
]


def test_annotate_entity_tokens():
    entity = message.data['entities'][0]
    expected = ['B-WebService', 'I-WebService']
    assert annotate_entity_tokens(entity) == expected


def test_merge_spans():
    expected = [(0, 11), (12, 14), (15, 33), (33, 34)]
    assert merge_spans(spans, entities) == expected


def test_merge_spans_advanced():
    test_spans = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    test_entities = [{'start': 1, 'end': 3}, {'start': 3, 'end': 6}]
    actual = merge_spans(test_spans, test_entities)
    expected = [(0, 1), (1, 3), (3, 6), (6, 7)]
    assert actual == expected


def test_annotate_tokens():
    expected = ['O', 'O', 'B-WebService', 'I-WebService', 'O']
    assert expected == annotate_tokens(spans, entities)


def test_convert_message_lines():
    expected = '\n'.join(lines)
    assert convert_message_lines(message) == expected


def test_convert_messages_lines():
    expected = '\n\n'.join(map(lambda i: '\n'.join(lines), range(3)))
    messages = [message for i in range(3)]
    assert convert_messages_lines(messages) == expected


def test_bug_omitted_entity():
    # i need a connection from [harras](StationStart) to [karl-preis-platz](StationDest) at [8 am](TimeStartTime).
    test_message = create_message(text='i need a connection from harras to karl-preis-platz at 8 am.',
                                  intent='Find Connection',
                                  entities=[
                                      {'end': 51, 'entity': 'StationDest', 'start': 35, 'value': 'karl-preis-platz'},
                                      {'end': 59, 'entity': 'TimeStartTime', 'start': 55, 'value': '8 am'},
                                      {'end': 31, 'entity': 'StationStart', 'start': 25, 'value': 'harras'}
                                  ],
                                  training=False,
                                  corpus=Corpus.MOCK)

    expected = '\n'.join([
        'i O',
        'need O',
        'a O',
        'connection O',
        'from O',
        'harras B-StationStart',
        'to O',
        'karl B-StationDest',
        '- I-StationDest',
        'preis I-StationDest',
        '- I-StationDest',
        'platz I-StationDest',
        'at O',
        '8 B-TimeStartTime',
        'am I-TimeStartTime',
        '. O'
    ])
    actual = convert_message_lines(test_message)
    assert actual == expected
