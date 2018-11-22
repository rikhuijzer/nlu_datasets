from src.converter import get_double_split, get_triple_split


def about_equal(a: int, b: int) -> bool:
    return abs(a - b) < 5


def test_get_double_split():
    x = list(range(100))
    y = ['A'] * 70 + ['B'] * 30
    train, test = get_double_split(x, y)
    assert about_equal(75, len(train.y))
    assert about_equal(25, len(test.y))


def test_get_triple_split():
    x = list(range(100))
    y = ['A'] * 70 + ['B'] * 30
    train, dev, test = get_triple_split(x, y)
    assert about_equal(60, len(train.y))
    assert about_equal(20, len(dev.y))
    assert about_equal(20, len(test.y))
