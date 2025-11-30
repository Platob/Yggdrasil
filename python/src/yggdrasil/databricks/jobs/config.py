import builtins
import dataclasses
import datetime as dt
import inspect
from dataclasses import dataclass, fields
from enum import Enum
from inspect import isclass
from typing import Any, Dict, List, get_type_hints, Optional, get_origin

from ...types.cast.registry import convert

__all__ = [
    "WidgetType",
    "NotebookConfig"
]

from yggdrasil.types.constants import SparkSession


def type_is_iterable(tpe: type, origin=None):
    if (
        tpe is list or tpe is set
    ):
        return True

    if origin is None:
        origin = get_origin(tpe)

        if origin is not None:
            return type_is_iterable(origin)
        return False
    else:
        return type_is_iterable(origin)


ALL_VALUES_TAG = "**all**"

class WidgetType(Enum):
    """Enum defining supported widget types in Databricks"""
    TEXT = "text"
    DROPDOWN = "dropdown"
    COMBOBOX = "combobox"
    MULTISELECT = "multiselect"
    DATETIME = "datetime"  # Custom handler for datetime (still uses text widget under the hood)


@dataclass
class NotebookConfig:
    @classmethod
    def get_dbutils(cls):
        # 1) explicit builtin injection (Databricks sometimes does this)
        if hasattr(builtins, "dbutils"):
            return builtins.dbutils

        # 2) ipython user namespace
        try:
            from IPython import get_ipython
            ip = get_ipython()
            if ip and getattr(ip, "user_ns", None):
                return ip.user_ns.get("dbutils")
        except Exception:
            pass

        # 3) caller globals (last resort, fragile)
        frame = inspect.currentframe()
        try:
            caller = frame.f_back
            for _ in range(3):
                if not caller:
                    break
                g = getattr(caller, "f_globals", None)
                if g and "dbutils" in g:
                    return g["dbutils"]
                caller = caller.f_back
        finally:
            del frame
            if 'caller' in locals():
                del caller

        return None

    @classmethod
    def from_environment(cls):
        """
        Build data class from environment variables or databricks dbutils if available.

        This method looks for values in the following order:
        1. Databricks widgets (if running in Databricks notebook)
        2. Databricks job parameters (if running as a job)
        3. Environment variables

        Returns:
            An instance of the dataclass populated with values from the environment
        """
        dbutils = cls.get_dbutils()
        key_values: Dict[str, Any] = {}

        # Get all field names and types for this dataclass
        class_fields = {field.name: field for field in fields(cls)}
        type_hints = get_type_hints(cls)

        for field_name, field in class_fields.items():
            field_value = None
            field_type = type_hints.get(field_name, Any)

            # Try to get value from Databricks widgets first (if available)
            if dbutils is not None:
                field_value = dbutils.widgets.get(field_name)

                if field_type is not str and type_is_iterable(field_type):
                    parts = field_value.split(",")
                    field_value = [
                        _ for
                        _ in parts
                        if _ != ALL_VALUES_TAG
                    ]

            key_values[field_name] = field_value

        # Convert the dict to a dataclass instance
        return convert(key_values, cls)

    @classmethod
    def _determine_widget_type(cls, field_type) -> WidgetType:
        """
        Determine the appropriate widget type for a given field type.

        Args:
            field_type: The type annotation of the field

        Returns:
            A WidgetType enum value representing the appropriate widget type
        """
        # Handle datetime types
        if field_type == dt.datetime or field_type == dt.date:
            return WidgetType.DATETIME

        # Handle enum types (use dropdown for these)
        if isclass(field_type):
            if issubclass(field_type, Enum):
                return WidgetType.DROPDOWN

        # Handle bool types (use dropdown with True/False options)
        if field_type == bool:
            return WidgetType.DROPDOWN

        # Handle list/set types that might benefit from multiselect
        origin = getattr(field_type, "__origin__", None)
        if origin in (list, List, set):
            return WidgetType.MULTISELECT

        # Default to text widget for all other types
        return WidgetType.TEXT

    @classmethod
    def _format_default_value(cls, value, widget_type: WidgetType) -> str:
        """
        Format a default value appropriately for the widget type.

        Args:
            value: The default value to format
            widget_type: The type of widget being used

        Returns:
            String representation of the value appropriate for the widget
        """
        if value is None:
            return ""

        # Handle datetime formatting
        if widget_type == WidgetType.DATETIME:
            if isinstance(value, dt.datetime):
                return value.isoformat()
            elif isinstance(value, dt.date):
                return value.strftime("%Y-%m-%d")

        # Handle multiselect (comma-separated values)
        if widget_type == WidgetType.MULTISELECT and isinstance(value, (list, tuple, set)):
            return ",".join(value)

        # Default: just convert to string
        return str(value)

    @classmethod
    def _widget_exists(cls, widget_name: str) -> bool:
        """
        Check if a widget with the given name already exists.

        Args:
            widget_name: Name of the widget to check

        Returns:
            True if the widget exists, False otherwise
        """
        dbutils = cls.get_dbutils()
        if dbutils is None or not hasattr(dbutils, "widgets"):
            return False

        try:
            # Try to get the widget value - will raise an exception if it doesn't exist
            dbutils.widgets.get(widget_name)
            return True
        except Exception:
            return False

    @classmethod
    def init_widgets(cls, skip_existing: bool = True):
        """
        Initialize Databricks widgets for each field in the dataclass.

        Args:
            skip_existing: If True, do not recreate widgets that already exist

        This method creates appropriate widgets for each field in the dataclass,
        with optional default values and customization options.
        """
        dbutils = cls.get_dbutils()
        if dbutils is None or not hasattr(dbutils, "widgets"):
            print("Widgets can only be initialized in a Databricks notebook environment")
            return

        # Get all field types
        type_hints = get_type_hints(cls)

        # Create widgets for each field
        for field in fields(cls):
            field_name = field.name
            field_type = type_hints.get(field_name, Any)

            # Skip this widget if it already exists and skip_existing is True
            if skip_existing and cls._widget_exists(field_name):
                continue

            # Otherwise infer from field type
            widget_type = cls._determine_widget_type(field_type)

            # Get default value
            if field.default != dataclasses.MISSING:
                default_value = field.default
            elif field.default_factory != dataclasses.MISSING:
                default_value = field.default_factory()
            else:
                default_value = []

            # For enum types, automatically use enum values if no options provided
            if field_type is bool:
                options = ["true", "false"]
            elif issubclass(field_type, Enum):
                options = [e.value for e in field_type]
            else:
                options = [
                    str(_.value) if isinstance(_, Enum) else str(_)
                    for _ in default_value
                ] if isinstance(default_value, (list, set, tuple)) else [str(default_value)]

            if not options:
                options = [ALL_VALUES_TAG]

            if widget_type in (WidgetType.DROPDOWN, WidgetType.COMBOBOX):
                # Create dropdown or combobox widget
                if widget_type == WidgetType.DROPDOWN:
                    dbutils.widgets.dropdown(
                        field_name, cls._format_default_value(options[0], widget_type),
                        options, field_name
                    )
                else:  # combobox
                    dbutils.widgets.combobox(
                        field_name, cls._format_default_value(options[0], widget_type),
                        options, field_name
                    )

            elif widget_type == WidgetType.MULTISELECT:
                dbutils.widgets.multiselect(
                    field_name, options[0],
                    options, field_name
                )

            else:  # TEXT and DATETIME both use text widgets
                dbutils.widgets.text(field_name, str(default_value) or "", field_name)

    @classmethod
    def init_job(cls):
        cls.init_widgets()

        if SparkSession is not None:
            spark_session = SparkSession.getActiveSession()

            if spark_session:
                spark_session.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
                spark_session.conf.set("spark.databricks.delta.merge.enableLowShuffle", "true")
                spark_session.conf.set("spark.databricks.delta.merge.optimizeInsertOnlyMerge", "true")
                spark_session.conf.set("spark.sql.session.timeZone", "UTC")

        return cls.from_environment()


class ExampleEnum(Enum):
    """Example enum for widget demonstration"""
    OPTION1 = "option1"
    OPTION2 = "option2"
    OPTION3 = "option3"


@dataclass
class CompleteNotebookConfig(NotebookConfig):
    """Example JobConfig with various field types to demonstrate widget handling"""
    # Basic types
    text_field: str
    integer_field: int = 42
    float_field: float = 3.14
    boolean_field: bool = True

    # Special types
    date_field: dt.date = dt.date(2023, 1, 1)
    datetime_field: dt.datetime = dt.datetime(2023, 1, 1, 12, 0, 0)
    enum_field: ExampleEnum = ExampleEnum.OPTION1

    # Collection types
    list_of_strings: List[str] = None  # Will be displayed as multiselect

    # Optional fields
    optional_text: Optional[str] = None
