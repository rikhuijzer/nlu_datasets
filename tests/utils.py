from typing import Iterable

from rasa_nlu.training_data.message import Message

from nlu_datasets.my_types import Corpus


def get_mock_messages(corpus: Corpus) -> Iterable[Message]:
    def create_mock_message(x: int) -> Message:
        from nlu_datasets.utils import create_message
        return create_message(text=str(x), intent='A' if 0 <= x < 10 else 'B', entities=[],
                              training=True if x < 15 else False, corpus=corpus)

    return map(create_mock_message, range(0, 20))
