TAG_PREFIX = b"t:"
DBX_META_PREFIX = b"databricks:"
DEFAULT_VALUE_KEY = b"default"
ALIAS_KEY = b"alias"
POSITION_KEY = b"position"
DEFAULT_FIELD_NAME = ""
# Field.media_type stores the on-disk MediaType under this metadata
# key as the canonical mime-string. Lives at the StructField/Schema
# level when stamped by FolderPath._persist_schema; the accessor
# itself lives on Field so per-column hints can ride the same slot.
MEDIA_TYPE_METADATA_KEY = b"media_type"
