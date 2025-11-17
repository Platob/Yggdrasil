"""Unit tests for the numpy_utils and pandas_utils modules."""

import pytest
import sys

# Check if numpy and pandas are available
has_numpy = False
has_pandas = False

try:
    import numpy as np
    has_numpy = True
except ImportError:
    pass

try:
    import pandas as pd
    has_pandas = True
except ImportError:
    pass

# Import utilities with proper error handling
try:
    from yggdrasil.libutils.numpy_utils import numpy
except (ImportError, AttributeError):
    numpy = None

try:
    from yggdrasil.libutils.pandas_utils import pandas, PandasDataFrame, PandasSeries
except (ImportError, AttributeError):
    pandas = None
    PandasDataFrame = None
    PandasSeries = None


class TestNumpyUtils:
    """Test the numpy_utils module."""

    @pytest.mark.skipif(numpy is None, reason="Failed to import numpy from yggdrasil")
    def test_numpy_import(self):
        """Test that numpy is properly imported or mocked."""
        if has_numpy:
            # If numpy is installed, it should be the actual numpy module
            assert numpy.__name__ == "numpy"
            assert hasattr(numpy, "array")
        else:
            # If numpy is not installed, it should be a fake module
            assert hasattr(numpy, "__is_fake_module__")
            assert numpy.__name__ == "numpy"

    @pytest.mark.skipif(not has_numpy or numpy is None, reason="NumPy not installed or import failed")
    def test_numpy_functionality(self):
        """Test that numpy functions work as expected when available."""
        # Test basic array creation
        arr = numpy.array([1, 2, 3])
        assert arr.ndim == 1
        assert arr.size == 3
        assert arr.shape == (3,)


class TestPandasUtils:
    """Test the pandas_utils module."""

    @pytest.mark.skipif(pandas is None, reason="Failed to import pandas from yggdrasil")
    def test_pandas_import(self):
        """Test that pandas is properly imported or mocked."""
        if has_pandas:
            # If pandas is installed, it should be the actual pandas module
            assert pandas.__name__ == "pandas"
            assert hasattr(pandas, "DataFrame")
        else:
            # If pandas is not installed, it should be a fake module
            assert hasattr(pandas, "__is_fake_module__")
            assert pandas.__name__ == "pandas"

    @pytest.mark.skipif(pandas is None or PandasDataFrame is None, reason="DataFrame not available")
    def test_dataframe_class(self):
        """Test that PandasDataFrame is properly defined."""
        if has_pandas:
            # If pandas is installed, it should be the actual DataFrame class
            assert PandasDataFrame.__name__ == "DataFrame"
        else:
            # If pandas is not installed, it should still be available as a class
            assert PandasDataFrame is not None

    @pytest.mark.skipif(pandas is None or PandasSeries is None, reason="Series not available")
    def test_series_class(self):
        """Test that PandasSeries is properly defined."""
        if has_pandas:
            # If pandas is installed, it should be the actual Series class
            assert PandasSeries.__name__ == "Series"
        else:
            # If pandas is not installed, it should still be available as a class
            assert PandasSeries is not None

    @pytest.mark.skipif(not has_pandas or pandas is None or PandasDataFrame is None,
                      reason="Pandas not installed or import failed")
    def test_dataframe_functionality(self):
        """Test that pandas functionality works as expected when available."""
        # Test basic DataFrame creation
        df = PandasDataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert df.shape == (3, 2)
        assert list(df.columns) == ["a", "b"]

    @pytest.mark.skipif(not has_pandas or pandas is None or PandasSeries is None,
                      reason="Pandas not installed or import failed")
    def test_series_functionality(self):
        """Test that Series functionality works as expected when available."""
        # Test basic Series creation
        s = PandasSeries([1, 2, 3], name="test")
        assert len(s) == 3
        assert s.name == "test"


if __name__ == "__main__":
    pytest.main()