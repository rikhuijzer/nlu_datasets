from src.utils import create_message
from src.my_types import Corpus
from src.converter import convert_message_lines, annotate_tokens, annotate_entity_tokens, merge_spans

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


def test_annotate_entity_tokens():
    entity = message.data['entities'][0]
    expected = ['B-WebService', 'I-WebService']
    assert expected == annotate_entity_tokens(entity)


def test_merge_spans():
    expected = [(0, 11), (12, 14), (15, 33), (33, 34)]
    assert expected == merge_spans(spans, entities)


def test_annotate_tokens():
    expected = ['O', 'O', 'B-WebService', 'I-WebService', 'O']
    assert expected == annotate_tokens(spans, entities)


def test_convert_message_lines():
    expected = [
        'Alternative O',
        'to O',
        'Facebook B-WebService',
        'Messenger I-WebService',
        '. O'
    ]
    expected = '\n'.join(expected)
    assert convert_message_lines(message) == expected


def test_to_ner_file():
    raise NotImplementedError()
