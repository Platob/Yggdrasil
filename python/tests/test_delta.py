from __future__ import annotations

import os
import pathlib
import shutil
import sys
import tempfile
from typing import Generator

import polars as pl
import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

try:
    import deltalake
    from yggdrasil.data import (
        DeltaReader, DeltaReaderConfig,
        DeltaWriter, DeltaWriterConfig,
        HAS_DELTA, ReadOptions, WriteOptions, WriteMode,
        eq
    )
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False


# Skip all tests if Delta Lake is not available
pytestmark = pytest.mark.skipif(
    not DELTA_AVAILABLE,
    reason="Delta Lake package not available. Install with 'pip install yggdrasil[delta]'",
)


@pytest.fixture
def temp_delta_dir() -> Generator[str, None, None]:
    """Create a temporary directory for Delta tables."""
    temp_dir = tempfile.mkdtemp(prefix="yggdrasil_delta_test_")
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
        "value": [10.5, 20.0, 30.5, 40.0, 50.5],
        "active": [True, False, True, True, False],
    })


@pytest.mark.skipif(not DELTA_AVAILABLE, reason="Delta Lake not available")
def test_delta_writer_creates_table(temp_delta_dir: str, sample_df: pl.DataFrame) -> None:
    """Test that DeltaWriter can create a new Delta table."""
    table_path = os.path.join(temp_delta_dir, "test_create")

    config = DeltaWriterConfig(table_path=table_path)
    writer = DeltaWriter(config)
    result = writer.from_polars(sample_df)

    assert result["status"] == "success"
    assert result["operation"] == "create"
    assert result["num_rows"] == 5
    assert result["num_columns"] == 4

    # Verify the table exists
    assert writer.exists() is True


@pytest.mark.skipif(not DELTA_AVAILABLE, reason="Delta Lake not available")
def test_delta_reader_reads_table(temp_delta_dir: str, sample_df: pl.DataFrame) -> None:
    """Test that DeltaReader can read a Delta table."""
    table_path = os.path.join(temp_delta_dir, "test_read")

    # First write the table
    config = DeltaWriterConfig(table_path=table_path)
    writer = DeltaWriter(config)
    writer.from_polars(sample_df)

    # Then read it back
    reader_config = DeltaReaderConfig(table_path=table_path)
    reader = DeltaReader(reader_config)
    df = reader.to_polars()

    # Check that the data matches
    assert len(df) == 5
    assert df.columns == ["id", "name", "value", "active"]

    # Compare actual data (convert to dicts for easier comparison)
    orig_data = sample_df.to_dicts()
    read_data = df.to_dicts()
    assert read_data == orig_data


@pytest.mark.skipif(not DELTA_AVAILABLE, reason="Delta Lake not available")
def test_delta_reader_with_filter(temp_delta_dir: str, sample_df: pl.DataFrame) -> None:
    """Test that DeltaReader can filter data."""
    table_path = os.path.join(temp_delta_dir, "test_filter")

    # First write the table
    config = DeltaWriterConfig(table_path=table_path)
    writer = DeltaWriter(config)
    writer.from_polars(sample_df)

    # Read with filter
    reader_config = DeltaReaderConfig(table_path=table_path)
    reader = DeltaReader(reader_config)
    options = ReadOptions(predicate=eq("active", True))
    df = reader.to_polars(options)

    # Check that only active records were returned
    assert len(df) == 3  # Alice, Charlie, Dave
    assert all(row["active"] for row in df.to_dicts())


@pytest.mark.skipif(not DELTA_AVAILABLE, reason="Delta Lake not available")
def test_delta_writer_append_mode(temp_delta_dir: str, sample_df: pl.DataFrame) -> None:
    """Test that DeltaWriter can append to an existing table."""
    table_path = os.path.join(temp_delta_dir, "test_append")

    # First write the initial data
    config = DeltaWriterConfig(table_path=table_path)
    writer = DeltaWriter(config)
    writer.from_polars(sample_df.slice(0, 2))  # Just first 2 rows

    # Then append more data
    append_options = WriteOptions(mode=WriteMode.APPEND)
    writer.from_polars(sample_df.slice(2, 3), append_options)  # Next 3 rows

    # Read it all back
    reader_config = DeltaReaderConfig(table_path=table_path)
    reader = DeltaReader(reader_config)
    df = reader.to_polars()

    # Should have all 5 rows
    assert len(df) == 5


@pytest.mark.skipif(not DELTA_AVAILABLE, reason="Delta Lake not available")
def test_delta_writer_overwrite_mode(temp_delta_dir: str, sample_df: pl.DataFrame) -> None:
    """Test that DeltaWriter can overwrite an existing table."""
    table_path = os.path.join(temp_delta_dir, "test_overwrite")

    # First write all data
    config = DeltaWriterConfig(table_path=table_path)
    writer = DeltaWriter(config)
    writer.from_polars(sample_df)

    # Then overwrite with just 2 rows
    new_data = pl.DataFrame({
        "id": [99, 100],
        "name": ["Xavier", "Yolanda"],
        "value": [99.9, 100.0],
        "active": [True, True],
    })

    overwrite_options = WriteOptions(mode=WriteMode.OVERWRITE)
    writer.from_polars(new_data, overwrite_options)

    # Read it back
    reader_config = DeltaReaderConfig(table_path=table_path)
    reader = DeltaReader(reader_config)
    df = reader.to_polars()

    # Should have only the 2 new rows
    assert len(df) == 2
    assert set(df["id"].to_list()) == {99, 100}


@pytest.mark.skipif(not DELTA_AVAILABLE, reason="Delta Lake not available")
def test_delta_reader_metadata(temp_delta_dir: str, sample_df: pl.DataFrame) -> None:
    """Test that DeltaReader can retrieve metadata."""
    table_path = os.path.join(temp_delta_dir, "test_metadata")

    # Write the table
    config = DeltaWriterConfig(table_path=table_path)
    writer = DeltaWriter(config)
    writer.from_polars(sample_df)

    # Get metadata
    reader_config = DeltaReaderConfig(table_path=table_path)
    reader = DeltaReader(reader_config)
    metadata = reader.get_metadata()

    assert metadata["table_uri"] == table_path
    assert metadata["version"] == 0  # First version
    assert len(metadata["schema"]["fields"]) == 4  # Four columns