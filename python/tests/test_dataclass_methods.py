from yggdrasil.dataclasses import dataclass
import dataclasses as py_dataclasses


@dataclass
class Person:
    name: str
    age: int
    nickname: str = "buddy"


def test_to_and_from_dict():
    person = Person("Alice", 30)

    assert person.to_dict() == {"name": "Alice", "age": 30, "nickname": "buddy"}

    recreated = Person.from_dict({"name": "Bob", "age": 25})

    assert recreated == Person("Bob", 25)

    defaulted = Person.from_dict({"name": "Carl"})

    assert defaulted == Person("Carl", 0, "buddy")


def test_to_and_from_tuple():
    person = Person("Charlie", 40, nickname="chuck")

    assert person.to_tuple() == ("Charlie", 40, "chuck")

    recreated = Person.from_tuple(["Dana", 35, "dee"])

    assert recreated == Person("Dana", 35, "dee")


def test_from_tuple_length_mismatch():
    try:
        Person.from_tuple(["Edgar", 28])
    except TypeError as exc:
        assert "Expected" in str(exc)
    else:
        raise AssertionError("Expected a TypeError when tuple length mismatches")


def test_standard_dataclasses_are_patched():
    @py_dataclasses.dataclass
    class Point:
        x: int
        y: int

    point = Point(1, 2)

    assert point.to_dict() == {"x": 1, "y": 2}
    assert Point.from_dict({"x": 3, "y": 4}) == Point(3, 4)


def test_default_instance():
    person = Person.default_instance()

    assert person == Person("", 0, "buddy")

    cached = Person.__default_instance__
    another = Person.default_instance()

    assert Person.__default_instance__ is cached
    assert another == person


def test_default_instance_standard_dataclass():
    @py_dataclasses.dataclass
    class WithDefaults:
        name: str
        count: int = 2

    instance = WithDefaults.default_instance()

    assert instance == WithDefaults("", 2)


def test_copy():
    person = Person.copy("Dora")

    assert person == Person("Dora", 0, "buddy")

    older = Person.copy("Erin", 42)

    assert older == Person("Erin", 42, "buddy")

    renamed = Person.copy("Finn", nickname="friend")

    assert renamed == Person("Finn", 0, "friend")

    try:
        Person.copy("Gale", 1, "extra", "values")
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError on too many positional args")
