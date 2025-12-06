import pytest

pa = pytest.importorskip("pyarrow")
pandas = pytest.importorskip("pandas")

from yggdrasil.types.cast.arrow_cast import CastOptions, arrow_schema_to_field
from yggdrasil.types.cast.pandas_cast import cast_pandas_dataframe, cast_pandas_series


# ---------------------------------------------------------------------------
# Series casting tests
# ---------------------------------------------------------------------------

def test_cast_pandas_series_struct_child_defaults_and_missing_added():
    series = pandas.Series([{"value": 1}, {"value": None}], name="payload")

    target_field = pa.field(
        "payload",
        pa.struct(
            [
                pa.field("value", pa.int64(), nullable=False),
                pa.field("label", pa.string(), nullable=False),
            ]
        ),
        nullable=True,
    )

    opts = CastOptions.safe_init(target_field=target_field, add_missing_columns=True)
    casted = cast_pandas_series(series, opts)

    assert isinstance(casted, pandas.Series)
    assert casted.name == series.name
    assert casted.tolist() == [
        {"value": 1, "label": ""},
        {"value": 0, "label": ""},
    ]


def test_cast_pandas_series_list_of_structs_preserves_null_lists():
    series = pandas.Series([[{"count": 1}], [{"count": None}], None], name="items")

    target_field = pa.field(
        "items",
        pa.list_(
            pa.field(
                "item",
                pa.struct([pa.field("count", pa.int64(), nullable=False)]),
                nullable=True,
            )
        ),
        nullable=True,
    )

    casted = cast_pandas_series(series, CastOptions.safe_init(target_field=target_field))

    assert isinstance(casted, pandas.Series)
    assert casted.tolist() == [
        [{"count": 1}],
        [{"count": 0}],
        None,
    ]


# ---------------------------------------------------------------------------
# DataFrame casting tests
# ---------------------------------------------------------------------------

def test_cast_pandas_dataframe_nested_schema_and_defaults():
    df = pandas.DataFrame(
        {
            "Meta": [{"id": 1}, {"id": 2}],
            "payload": [{"score": "7"}, {"score": None}],
            "extra": [True, False],
        }
    )

    target_schema = pa.schema(
        [
            pa.field(
                "meta",
                pa.struct(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("tag", pa.string(), nullable=False),
                    ]
                ),
                nullable=False,
            ),
            pa.field(
                "payload",
                pa.struct([pa.field("score", pa.int64(), nullable=False)]),
                nullable=False,
            ),
            pa.field("extra", pa.bool_(), nullable=True),
        ]
    )

    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(target_schema),
        add_missing_columns=True,
        allow_add_columns=False,
        strict_match_names=False,
    )

    casted = cast_pandas_dataframe(df, opts)

    assert list(casted.columns) == ["meta", "payload", "extra"]
    assert casted["meta"].tolist() == [
        {"id": 1, "tag": ""},
        {"id": 2, "tag": ""},
    ]
    assert casted["payload"].tolist() == [
        {"score": 7},
        {"score": 0},
    ]
    assert casted["extra"].tolist() == [True, False]


def test_cast_pandas_dataframe_preserves_extras_when_allowed():
    df = pandas.DataFrame({"a": [1], "keep": [9]})

    opts = CastOptions.safe_init(
        target_field=arrow_schema_to_field(pa.schema([pa.field("a", pa.int64(), nullable=False)])),
        allow_add_columns=True,
    )

    casted = cast_pandas_dataframe(df, opts)

    assert list(casted.columns) == ["a", "keep"]
    assert casted["a"].tolist() == [1]
    assert casted["keep"].tolist() == [9]
