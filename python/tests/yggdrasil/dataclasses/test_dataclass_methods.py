import dataclasses as py_dataclasses
import pyarrow as pa

from yggdrasil.dataclasses import dataclass


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


def test_arrow_field_method():
    field = Person.arrow_field()

    assert field.name == "Person"
    assert field.type == pa.struct(
        [
            pa.field("name", pa.string(), nullable=False),
            pa.field("age", pa.int64(), nullable=False),
            pa.field("nickname", pa.string(), nullable=False),
        ]
    )
    assert field.nullable is False


def test_arrow_field_name_override_and_standard_patch():
    field = Person.arrow_field(name="person_record")

    assert field.name == "person_record"

    @py_dataclasses.dataclass
    class Point:
        x: int
        y: int

    point_field = Point.arrow_field()

    assert point_field.type == pa.struct(
        [
            pa.field("x", pa.int64(), nullable=False),
            pa.field("y", pa.int64(), nullable=False),
        ]
    )


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


def test_safe_init_casts_and_defaults():
    person = Person.safe_init("Hank", "34")

    assert person == Person("Hank", 34, "buddy")

    with_defaults = Person.safe_init("Ivy")

    assert with_defaults == Person("Ivy", 0, "buddy")


def test_safe_init_rejects_invalid_fields():
    try:
        Person.safe_init("Jake", unknown=1)
    except TypeError as exc:
        assert "invalid field" in str(exc)
    else:
        raise AssertionError("Expected TypeError for invalid field overrides")
