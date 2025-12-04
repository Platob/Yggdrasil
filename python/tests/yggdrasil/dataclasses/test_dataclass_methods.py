from yggdrasil.dataclasses import yggdataclass


@yggdataclass
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


def test_from_dict_safe_casting():
    recreated = Person.from_dict({"name": "Bob", "age": "25"})

    assert recreated == Person("Bob", 25)

    empty_age = Person.from_dict({"name": "Dana", "age": ""})

    assert empty_age == Person("Dana", 0, "buddy")

    unsafe = Person.from_dict({"name": "Eve", "age": "28"}, safe=False)

    assert unsafe.age == "28"


def test_to_and_from_tuple():
    person = Person("Charlie", 40, nickname="chuck")

    assert person.to_tuple() == ("Charlie", 40, "chuck")

    recreated = Person.from_tuple(["Dana", 35, "dee"])

    assert recreated == Person("Dana", 35, "dee")


def test_from_tuple_safe_casting():
    recreated = Person.from_tuple(["Fay", "41", "friend"])

    assert recreated == Person("Fay", 41, "friend")

    unsafe = Person.from_tuple(["Gil", "42", "g"], safe=False)

    assert unsafe.age == "42"


def test_from_tuple_length_mismatch():
    try:
        Person.from_tuple(["Edgar", 28])
    except TypeError as exc:
        assert "Expected" in str(exc)
    else:
        raise AssertionError("Expected a TypeError when tuple length mismatches")
