from __future__ import annotations

import fnmatch
import io
import ipaddress
import os
import re
import struct
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import ClassVar, Generic, Mapping

from yggdrasil.pickle.ser.serialized import Serialized, T
from yggdrasil.pickle.ser.tags import Tags
from yggdrasil.io.url import URL

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


__all__ = [
    "MAX_INLINE_DIR_BYTES",
    "LogicalSerialized",
    "DecimalSerialized",
    "DatetimeSerialized",
    "DateSerialized",
    "TimeSerialized",
    "TimedeltaSerialized",
    "TimezoneSerialized",
    "UUIDSerialized",
    "ComplexNumberSerialized",
    "BytesSerialized",
    "PathSerialized",
    "IPAddressSerialized",
]


# ============================================================================
# constants
# ============================================================================

EPOCH_DATE = date(1970, 1, 1)
EPOCH_DATETIME = datetime(1970, 1, 1, tzinfo=UTC)
DAY_MICROS = 86_400 * 1_000_000

MAX_INLINE_DIR_BYTES = 1024 * 1024  # 1 MiB

# compact metadata keys
M_SCALE = b"s"     # decimal scale
M_PREC = b"p"      # decimal precision
M_UNIT = b"u"      # time/datetime/timedelta unit
M_TZ = b"z"        # timezone text
M_KIND = b"k"      # subtype discriminator

# compact path metadata keys
M_PATH_MODE = b"pm"   # path mode: p / f / dz
M_PATH_OS = b"po"     # source os: windows / posix
M_FILENAME = b"fn"    # original filename for file payloads
M_DIRNAME = b"dn"     # original dirname for directory payloads

# metadata values
PATH_MODE_PATH = b"p"
PATH_MODE_FILE = b"f"
PATH_MODE_DIR_ZIP = b"dz"

OS_WINDOWS = b"windows"
OS_POSIX = b"posix"

K_BYTES = b"b"
K_BYTEARRAY = b"ba"
K_MEMORYVIEW = b"mv"

K_PATH_GENERIC = b"p"
K_PATH_POSIX = b"pp"
K_PATH_WINDOWS = b"pw"

K_TZ = b"tz"

K_IP4_ADDR = b"4a"
K_IP6_ADDR = b"6a"
K_IP4_IFACE = b"4i"
K_IP6_IFACE = b"6i"
K_IP4_NET = b"4n"
K_IP6_NET = b"6n"

U_S = "s"
U_MS = "ms"
U_US = "us"
U_NS = "ns"


# ============================================================================
# metadata helpers
# ============================================================================

def _metadata_merge(
    base: Mapping[bytes, bytes] | None,
    extra: Mapping[bytes, bytes] | None = None,
) -> dict[bytes, bytes] | None:
    if not base and not extra:
        return None

    out: dict[bytes, bytes] = {}
    if base:
        out.update(base)
    if extra:
        out.update(extra)
    return out


def _metadata_bytes(
    metadata: Mapping[bytes, bytes] | None,
    key: bytes,
) -> bytes | None:
    if not metadata:
        return None
    return metadata.get(key)


def _metadata_text(
    metadata: Mapping[bytes, bytes] | None,
    key: bytes,
    default: str | None = None,
) -> str | None:
    raw = _metadata_bytes(metadata, key)
    if raw is None:
        return default
    return raw.decode("utf-8")


def _metadata_int(
    metadata: Mapping[bytes, bytes] | None,
    key: bytes,
    default: int | None = None,
) -> int | None:
    raw = _metadata_bytes(metadata, key)
    if raw is None:
        return default
    return int(raw.decode("ascii"))


# ============================================================================
# binary packing helpers
# ============================================================================

def _pack_i32(value: int) -> bytes:
    return struct.pack(">i", value)


def _pack_i64(value: int) -> bytes:
    return struct.pack(">q", value)


def _pack_u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _pack_f64_pair(a: float, b: float) -> bytes:
    return struct.pack(">dd", a, b)


def _unpack_i32(data: bytes, *, tag_name: str) -> int:
    if len(data) != 4:
        raise ValueError(f"{tag_name} payload must be exactly 4 bytes, got {len(data)}")
    return int(struct.unpack(">i", data)[0])


def _unpack_i64(data: bytes, *, tag_name: str) -> int:
    if len(data) != 8:
        raise ValueError(f"{tag_name} payload must be exactly 8 bytes, got {len(data)}")
    return int(struct.unpack(">q", data)[0])


def _unpack_u64(data: bytes, *, tag_name: str) -> int:
    if len(data) != 8:
        raise ValueError(f"{tag_name} payload must be exactly 8 bytes, got {len(data)}")
    return int(struct.unpack(">Q", data)[0])


def _unpack_f64_pair(data: bytes, *, tag_name: str) -> tuple[float, float]:
    if len(data) != 16:
        raise ValueError(f"{tag_name} payload must be exactly 16 bytes, got {len(data)}")
    a, b = struct.unpack(">dd", data)
    return float(a), float(b)


# ============================================================================
# unit conversion helpers
# ============================================================================

def _unit_to_micros(value: int, unit: str, *, tag_name: str) -> int:
    if unit == U_S:
        return value * 1_000_000
    if unit == U_MS:
        return value * 1_000
    if unit == U_US:
        return value
    if unit == U_NS:
        return value // 1_000
    raise ValueError(f"Unsupported {tag_name} unit: {unit!r}")


def _datetime_from_epoch(value: int, unit: str) -> datetime:
    if unit == U_S:
        return datetime.fromtimestamp(value, tz=UTC)
    if unit == U_MS:
        return EPOCH_DATETIME + timedelta(milliseconds=value)
    if unit == U_US:
        return EPOCH_DATETIME + timedelta(microseconds=value)
    if unit == U_NS:
        return EPOCH_DATETIME + timedelta(microseconds=value // 1_000)
    raise ValueError(f"Unsupported DATETIME unit: {unit!r}")


def _timedelta_from_value(value: int, unit: str) -> timedelta:
    if unit == U_S:
        return timedelta(seconds=value)
    if unit == U_MS:
        return timedelta(milliseconds=value)
    if unit == U_US:
        return timedelta(microseconds=value)
    if unit == U_NS:
        return timedelta(microseconds=value // 1_000)
    raise ValueError(f"Unsupported TIMEDELTA unit: {unit!r}")


def _time_from_offset(value: int, unit: str) -> time:
    if value < 0:
        raise ValueError("TIME payload must be >= 0")

    total_micros = _unit_to_micros(value, unit, tag_name="TIME")
    if total_micros >= DAY_MICROS:
        raise ValueError("TIME payload must be less than one day")

    hour, rem = divmod(total_micros, 3_600_000_000)
    minute, rem = divmod(rem, 60_000_000)
    second, microsecond = divmod(rem, 1_000_000)

    return time(
        hour=int(hour),
        minute=int(minute),
        second=int(second),
        microsecond=int(microsecond),
    )


# ============================================================================
# timezone helpers
# ============================================================================

def _tz_to_text(tz: tzinfo | None, ref_dt: datetime | None = None) -> str | None:
    if tz is None:
        return None

    if tz is UTC:
        return "UTC"

    if ZoneInfo is not None and isinstance(tz, ZoneInfo):
        key = getattr(tz, "key", None)
        if isinstance(key, str) and key:
            return key

    offset = tz.utcoffset(ref_dt)
    if offset is not None:
        total_seconds = int(offset.total_seconds())
        sign = "+" if total_seconds >= 0 else "-"
        total_seconds = abs(total_seconds)
        hh, rem = divmod(total_seconds, 3600)
        mm = rem // 60
        return f"{sign}{hh:02d}:{mm:02d}"

    name = tz.tzname(ref_dt)
    if name:
        return name

    return None


def _load_tzinfo(metadata: Mapping[bytes, bytes] | None) -> tzinfo | None:
    tz_name = _metadata_text(metadata, M_TZ)
    if not tz_name:
        return None

    if tz_name == "UTC":
        return UTC

    if tz_name.startswith(("+", "-")):
        sign = 1 if tz_name[0] == "+" else -1
        hh, mm = tz_name[1:].split(":", 1)
        delta = timedelta(hours=int(hh), minutes=int(mm))
        return timezone(sign * delta)

    if ZoneInfo is not None:
        try:
            return ZoneInfo(tz_name)
        except Exception:
            pass

    raise ValueError(f"Invalid or unsupported timezone metadata: {tz_name!r}")


# ============================================================================
# path helpers
# ============================================================================

def _current_os_name() -> bytes:
    """
    Use broad path semantics instead of kernel branding.
    """
    return OS_WINDOWS if os.name == "nt" else OS_POSIX


def _sanitize_path_part(part: str) -> str:
    """
    Keep translated cross-OS path segments safe-ish.

    This is used only when translating a Windows absolute path into a POSIX path.
    """
    part = part.replace("\\", "_").replace("/", "_").replace(":", "_")
    part = re.sub(r"[\x00-\x1f]", "_", part)
    return part or "_"


def _rebuild_path(raw: str, *, source_os: bytes) -> Path:
    """
    Rebuild a serialized path on the current runtime.

    Same OS family:
        - preserve through Path(raw)

    Cross-OS:
        - windows -> posix:
            C:\\tmp\\x -> /c/tmp/x
            tmp\\x     -> tmp/x
        - posix -> windows:
            /var/tmp/x -> \\var\\tmp\\x
            var/tmp/x  -> var\\tmp\\x
    """
    current_os = _current_os_name()

    if source_os == current_os:
        return Path(raw)

    if source_os == OS_POSIX:
        p = PurePosixPath(raw)

        if current_os == OS_WINDOWS:
            parts = [part for part in p.parts if part != "/"]
            if p.is_absolute():
                if not parts:
                    return Path("\\")
                return Path("\\" + "\\".join(parts))
            return Path("\\".join(parts))

        return Path(raw)

    if source_os == OS_WINDOWS:
        p = PureWindowsPath(raw)

        if current_os == OS_POSIX:
            if not p.is_absolute() and not p.drive:
                parts = [part for part in p.parts if part not in ("\\", "/")]
                return Path("/".join(parts))

            drive = p.drive.rstrip(":").lower() if p.drive else "_"
            drive = _sanitize_path_part(drive)
            tail_parts = [
                _sanitize_path_part(part)
                for part in p.parts
                if part not in ("\\", "/", p.anchor)
            ]
            translated = "/" + "/".join([drive, *tail_parts]) if tail_parts else f"/{drive}"
            return Path(translated)

        return Path(raw)

    return Path(raw)


def _dir_total_bytes(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def _should_exclude_path(parts: tuple[str, ...]) -> bool:
    """
    Skip noisy cache/build/junk paths when embedding directories inline.
    """
    exact = {
        "__pycache__",
        "__pypackages__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".hypothesis",
        ".tox",
        ".nox",
        ".coverage",
        "htmlcov",
        ".eggs",
        ".ipynb_checkpoints",

        ".git",
        ".svn",
        ".hg",
        ".bzr",

        ".venv",
        "venv",
        "env",
        ".env",

        "node_modules",
        ".pnpm-store",
        ".npm",
        ".yarn",
        ".yarn-cache",
        ".yarnrc",
        ".parcel-cache",
        ".next",
        ".nuxt",
        ".svelte-kit",
        ".turbo",
        ".angular",
        ".cache-loader",
        ".eslintcache",
        ".stylelintcache",

        "target",
        "cmake-build-debug",
        "cmake-build-release",
        "cmake-build-relwithdebinfo",
        "cmake-build-minsizerel",
        "CMakeFiles",
        "CMakeCache.txt",
        "compile_commands.json",
        ".cache",
        ".ccls-cache",
        ".clangd",

        ".gradle",
        ".mvn",
        "out",

        ".idea",
        ".vscode",
        ".vs",
        ".fleet",

        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",

        ".dvc",
        ".snakemake",
        ".meltano",
        ".dbx",
        ".databricks",
        ".terraform",
        ".terragrunt-cache",

        "build",
        "dist",
        "coverage",
        "tmp",
        "temp",
    }

    patterns = {
        "*.egg-info",
        "*.dist-info",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.dylib",
        "*.dll",
        "*.o",
        "*.obj",
        "*.a",
        "*.lib",
        "*.class",
    }

    for part in parts:
        if part in exact:
            return True
        if any(fnmatch.fnmatch(part, pattern) for pattern in patterns):
            return True

    return False


def _zip_directory_to_bytes(root: Path) -> bytes:
    """
    Zip a directory into bytes.
    """
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if not p.exists():
                continue

            arcname = p.relative_to(root)

            if _should_exclude_path(arcname.parts):
                continue

            if p.is_dir():
                if arcname.parts:
                    zf.writestr(f"{arcname.as_posix().rstrip('/')}/", b"")
            elif p.is_file():
                zf.write(filename=p, arcname=arcname.as_posix())

    return buf.getvalue()


def _extract_zip_bytes_to_temp_dir(data: bytes, *, dirname: str) -> Path:
    tmp_root = Path(tempfile.mkdtemp(prefix="ygg_path_dir_"))
    out_dir = tmp_root / dirname
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(data), mode="r") as zf:
        zf.extractall(out_dir)

    return out_dir


def _write_file_bytes_to_temp_path(data: bytes, *, filename: str) -> Path:
    suffix = Path(filename).suffix
    fd, name = tempfile.mkstemp(prefix="ygg_path_file_", suffix=suffix)
    os.close(fd)
    path = Path(name)
    path.write_bytes(data)
    return path

def _pack_i64_triplet(a: int, b: int, c: int) -> bytes:
    return struct.pack(">qqq", a, b, c)


def _unpack_i64_triplet(data: bytes, *, tag_name: str) -> tuple[int, int, int]:
    if len(data) != 24:
        raise ValueError(f"{tag_name} payload must be exactly 24 bytes, got {len(data)}")
    a, b, c = struct.unpack(">qqq", data)
    return int(a), int(b), int(c)

# ============================================================================
# base class
# ============================================================================

@dataclass(frozen=True, slots=True)
class LogicalSerialized(Serialized[T], Generic[T]):
    """
    Base class for compact logical / semantic Python payloads.

    Strategy:
    - use compact, fixed-width payload bytes whenever possible
    - use short metadata keys
    - keep decoding logic explicit and boring
    """

    TAG: ClassVar[int]

    @property
    def value(self) -> T:
        raise NotImplementedError

    def as_python(self) -> T:
        return self.value

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        # ------------------------------------------------------------------
        # Decimal
        # payload: int64 coefficient
        # meta: s=scale, p=precision(optional)
        # ------------------------------------------------------------------
        if isinstance(obj, Decimal):
            sign, digits, exponent = obj.as_tuple()
            coefficient = int("".join(map(str, digits))) if digits else 0
            if sign:
                coefficient = -coefficient

            precision = len(digits)
            scale = -exponent

            for name, value in (
                    ("coefficient", coefficient),
                    ("precision", precision),
                    ("scale", scale),
            ):
                if not (-0x8000000000000000 <= value <= 0x7FFFFFFFFFFFFFFF):
                    raise OverflowError(f"Decimal {name} does not fit int64 payload")

            return Serialized.build(
                tag=Tags.DECIMAL,
                data=_pack_i64_triplet(coefficient, precision, scale),
                metadata=metadata,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # datetime
        # payload: int64 epoch_us
        # meta: u, z
        # ------------------------------------------------------------------
        if isinstance(obj, datetime):
            merged = _metadata_merge(metadata) or {}

            if obj.tzinfo is None:
                dt_utc = obj.replace(tzinfo=UTC)
            else:
                dt_utc = obj.astimezone(UTC)
                tz_text = _tz_to_text(obj.tzinfo, obj)
                if tz_text:
                    merged.setdefault(M_TZ, tz_text.encode("utf-8"))

            epoch_us = int(dt_utc.timestamp() * 1_000_000)
            merged.setdefault(M_UNIT, U_US.encode("ascii"))

            return Serialized.build(
                tag=Tags.DATETIME,
                data=_pack_i64(epoch_us),
                metadata=merged,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # date
        # payload: int32 days since epoch
        # ------------------------------------------------------------------
        if isinstance(obj, date) and not isinstance(obj, datetime):
            days = (obj - EPOCH_DATE).days
            return Serialized.build(
                tag=Tags.DATE,
                data=_pack_i32(days),
                metadata=metadata,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # time
        # payload: uint64 offset from midnight in us
        # meta: u, z
        # ------------------------------------------------------------------
        if isinstance(obj, time):
            merged = _metadata_merge(metadata) or {}
            total_us = (
                ((obj.hour * 60 + obj.minute) * 60 + obj.second) * 1_000_000
                + obj.microsecond
            )

            tz_text = _tz_to_text(obj.tzinfo, None)
            if tz_text:
                merged.setdefault(M_TZ, tz_text.encode("utf-8"))
            merged.setdefault(M_UNIT, U_US.encode("ascii"))

            return Serialized.build(
                tag=Tags.TIME,
                data=_pack_u64(total_us),
                metadata=merged,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # timedelta
        # payload: int64 microseconds
        # meta: u
        # ------------------------------------------------------------------
        if isinstance(obj, timedelta):
            merged = _metadata_merge(metadata, {M_UNIT: U_US.encode("ascii")})
            total_us = (
                obj.days * 86_400_000_000
                + obj.seconds * 1_000_000
                + obj.microseconds
            )

            return Serialized.build(
                tag=Tags.TIMEDELTA,
                data=_pack_i64(total_us),
                metadata=merged,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # tzinfo / timezone / ZoneInfo
        # payload: utf-8 zone text
        # meta: k=tz
        # ------------------------------------------------------------------
        if isinstance(obj, tzinfo):
            tz_text = _tz_to_text(obj, None)
            if not tz_text:
                raise ValueError(f"Cannot serialize timezone object {obj!r}")

            merged = _metadata_merge(metadata, {M_KIND: K_TZ})
            return Serialized.build(
                tag=Tags.TIMEZONE,
                data=tz_text.encode("utf-8"),
                metadata=merged,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # uuid
        # payload: 16 raw bytes
        # ------------------------------------------------------------------
        if isinstance(obj, uuid.UUID):
            return Serialized.build(
                tag=Tags.UUID,
                data=obj.bytes,
                metadata=metadata,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # complex
        # payload: float64 real + float64 imag
        # ------------------------------------------------------------------
        if isinstance(obj, complex):
            return Serialized.build(
                tag=Tags.COMPLEX,
                data=_pack_f64_pair(obj.real, obj.imag),
                metadata=metadata,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # bytes-like
        # payload: raw bytes
        # meta: k=b|ba|mv
        # ------------------------------------------------------------------
        if isinstance(obj, bytes):
            return Serialized.build(
                tag=Tags.BYTES,
                data=obj,
                metadata=_metadata_merge(metadata, {M_KIND: K_BYTES}),
                codec=codec,
            )

        if isinstance(obj, bytearray):
            return Serialized.build(
                tag=Tags.BYTES,
                data=bytes(obj),
                metadata=_metadata_merge(metadata, {M_KIND: K_BYTEARRAY}),
                codec=codec,
            )

        if isinstance(obj, memoryview):
            return Serialized.build(
                tag=Tags.BYTES,
                data=obj.tobytes(),
                metadata=_metadata_merge(metadata, {M_KIND: K_MEMORYVIEW}),
                codec=codec,
            )

        # ------------------------------------------------------------------
        # path-like
        # uses PathSerialized logic
        # ------------------------------------------------------------------
        if isinstance(obj, Path):
            return PathSerialized.from_value(obj, metadata=metadata, codec=codec)

        if isinstance(obj, PureWindowsPath):
            return Serialized.build(
                tag=Tags.PATH,
                data=str(obj).encode("utf-8"),
                metadata=_metadata_merge(metadata, {
                    M_PATH_MODE: PATH_MODE_PATH,
                    M_PATH_OS: OS_WINDOWS,
                    M_KIND: K_PATH_WINDOWS,
                }),
                codec=codec,
            )

        if isinstance(obj, PurePosixPath):
            return Serialized.build(
                tag=Tags.PATH,
                data=str(obj).encode("utf-8"),
                metadata=_metadata_merge(metadata, {
                    M_PATH_MODE: PATH_MODE_PATH,
                    M_PATH_OS: OS_POSIX,
                    M_KIND: K_PATH_POSIX,
                }),
                codec=codec,
            )

        if isinstance(obj, PurePath):
            return Serialized.build(
                tag=Tags.PATH,
                data=str(obj).encode("utf-8"),
                metadata=_metadata_merge(metadata, {
                    M_PATH_MODE: PATH_MODE_PATH,
                    M_PATH_OS: _current_os_name(),
                    M_KIND: K_PATH_GENERIC,
                }),
                codec=codec,
            )

        # ------------------------------------------------------------------
        # url-like
        # ------------------------------------------------------------------

        if isinstance(obj, URL):
            return Serialized.build(
                tag=Tags.URL,
                data=obj.to_string(encode=True).encode("utf-8"),
                metadata=metadata,
                codec=codec,
            )

        # ------------------------------------------------------------------
        # ipaddress family
        # payload: canonical string
        # meta: k subtype
        # ------------------------------------------------------------------
        if isinstance(obj, ipaddress.IPv4Interface):
            kind = K_IP4_IFACE
        elif isinstance(obj, ipaddress.IPv6Interface):
            kind = K_IP6_IFACE
        elif isinstance(obj, ipaddress.IPv4Network):
            kind = K_IP4_NET
        elif isinstance(obj, ipaddress.IPv6Network):
            kind = K_IP6_NET
        elif isinstance(obj, ipaddress.IPv4Address):
            kind = K_IP4_ADDR
        elif isinstance(obj, ipaddress.IPv6Address):
            kind = K_IP6_ADDR
        else:
            kind = None

        if kind is not None:
            return Serialized.build(
                tag=Tags.IPADDRESS,
                data=str(obj).encode("utf-8"),
                metadata=_metadata_merge(metadata, {M_KIND: kind}),
                codec=codec,
            )

        return None


# ============================================================================
# logical subclasses
# ============================================================================

@dataclass(frozen=True, slots=True)
class DecimalSerialized(LogicalSerialized[Decimal]):
    """
    Decimal payload:
        payload = 3 x signed int64:
            coefficient
            precision
            scale

    Notes
    -----
    - coefficient is the signed integer formed from the decimal digits
    - precision is the number of digits
    - scale is the number of fractional decimal digits
    """

    TAG: ClassVar[int] = Tags.DECIMAL

    @property
    def value(self) -> Decimal:
        coefficient, precision, scale = _unpack_i64_triplet(
            self.decode(),
            tag_name="DECIMAL",
        )

        value = Decimal(coefficient).scaleb(-scale)
        return value


@dataclass(frozen=True, slots=True)
class DatetimeSerialized(LogicalSerialized[datetime]):
    """
    Datetime payload:
        payload = big-endian int64 epoch integer
        metadata:
            u = s|ms|us|ns
            z = optional timezone text
    """

    TAG: ClassVar[int] = Tags.DATETIME

    @property
    def value(self) -> datetime:
        unit = _metadata_text(self.metadata, M_UNIT, U_US) or U_US
        epoch_value = _unpack_i64(self.decode(), tag_name="DATETIME")
        dt = _datetime_from_epoch(epoch_value, unit)

        tz = _load_tzinfo(self.metadata)
        if tz is not None:
            dt = dt.astimezone(tz)

        return dt


@dataclass(frozen=True, slots=True)
class DateSerialized(LogicalSerialized[date]):
    """
    Date payload:
        payload = signed int32 days since epoch
    """

    TAG: ClassVar[int] = Tags.DATE

    @property
    def value(self) -> date:
        days = _unpack_i32(self.decode(), tag_name="DATE")
        return EPOCH_DATE + timedelta(days=days)


@dataclass(frozen=True, slots=True)
class TimeSerialized(LogicalSerialized[time]):
    """
    Time payload:
        payload = unsigned int64 offset from midnight
        metadata:
            u = s|ms|us|ns
            z = optional timezone text
    """

    TAG: ClassVar[int] = Tags.TIME

    @property
    def value(self) -> time:
        unit = _metadata_text(self.metadata, M_UNIT, U_US) or U_US
        raw_value = _unpack_u64(self.decode(), tag_name="TIME")
        t = _time_from_offset(raw_value, unit)

        tz = _load_tzinfo(self.metadata)
        if tz is not None:
            t = t.replace(tzinfo=tz)

        return t


@dataclass(frozen=True, slots=True)
class TimedeltaSerialized(LogicalSerialized[timedelta]):
    """
    Timedelta payload:
        payload = signed int64 duration
        metadata:
            u = s|ms|us|ns
    """

    TAG: ClassVar[int] = Tags.TIMEDELTA

    @property
    def value(self) -> timedelta:
        unit = _metadata_text(self.metadata, M_UNIT, U_US) or U_US
        raw_value = _unpack_i64(self.decode(), tag_name="TIMEDELTA")
        return _timedelta_from_value(raw_value, unit)


@dataclass(frozen=True, slots=True)
class TimezoneSerialized(LogicalSerialized[tzinfo]):
    """
    Timezone payload:
        payload = utf-8 timezone text
        metadata:
            k = tz
    """

    TAG: ClassVar[int] = Tags.TIMEZONE

    @property
    def value(self) -> tzinfo:
        kind = _metadata_bytes(self.metadata, M_KIND)
        if kind not in (None, K_TZ):
            raise ValueError(f"Invalid TIMEZONE metadata kind: {kind!r}")

        text = self.decode().decode("utf-8")
        tz = _load_tzinfo({M_TZ: text.encode("utf-8")})
        if tz is None:
            raise ValueError("TIMEZONE payload decoded to None")
        return tz


@dataclass(frozen=True, slots=True)
class UUIDSerialized(LogicalSerialized[uuid.UUID]):
    """
    UUID payload:
        payload = 16 raw bytes
    """

    TAG: ClassVar[int] = Tags.UUID

    @property
    def value(self) -> uuid.UUID:
        raw = self.decode()
        if len(raw) != 16:
            raise ValueError(f"UUID payload must be 16 bytes, got {len(raw)}")
        return uuid.UUID(bytes=raw)


@dataclass(frozen=True, slots=True)
class ComplexNumberSerialized(LogicalSerialized[complex]):
    """
    Complex payload:
        payload = float64(real) + float64(imag)
    """

    TAG: ClassVar[int] = Tags.COMPLEX

    @property
    def value(self) -> complex:
        real, imag = _unpack_f64_pair(self.decode(), tag_name="COMPLEX")
        return complex(real, imag)


@dataclass(frozen=True, slots=True)
class BytesSerialized(LogicalSerialized[bytes | bytearray | memoryview]):
    """
    Bytes payload:
        payload = raw bytes
        metadata:
            k = b | ba | mv
    """

    TAG: ClassVar[int] = Tags.BYTES

    @property
    def value(self) -> bytes | bytearray | memoryview:
        raw = self.decode()
        kind = _metadata_bytes(self.metadata, M_KIND)

        if kind in (None, K_BYTES):
            return raw
        if kind == K_BYTEARRAY:
            return bytearray(raw)
        if kind == K_MEMORYVIEW:
            return memoryview(raw)

        raise ValueError(f"Invalid BYTES metadata kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class PathSerialized(LogicalSerialized[Path]):
    """
    Path payload modes:

    1. path mode
       - metadata:
           pm = p
           po = source os family
       - payload:
           utf-8 string path
       - decode:
           rebuild path for current OS semantics

    2. file mode
       - metadata:
           pm = f
           fn = original filename
           po = source os family
       - payload:
           raw file bytes
       - decode:
           write to a temp file and return that path

    3. inline zipped directory mode
       - metadata:
           pm = dz
           dn = original directory name
           po = source os family
       - payload:
           zip bytes
       - decode:
           extract to temp dir and return that path

    Directory embedding is only used when total file bytes are below
    MAX_INLINE_DIR_BYTES.
    """

    TAG: ClassVar[int] = Tags.PATH

    @property
    def value(self) -> Path:
        metadata = self.metadata or {}
        mode = metadata.get(M_PATH_MODE, PATH_MODE_PATH)

        if mode == PATH_MODE_PATH:
            raw = self.decode().decode("utf-8")
            source_os = metadata.get(M_PATH_OS, OS_POSIX)
            return _rebuild_path(raw, source_os=source_os)

        if mode == PATH_MODE_FILE:
            filename = metadata.get(M_FILENAME, b"payload.bin").decode("utf-8", errors="replace")
            return _write_file_bytes_to_temp_path(self.decode(), filename=filename)

        if mode == PATH_MODE_DIR_ZIP:
            dirname = metadata.get(M_DIRNAME, b"directory").decode("utf-8", errors="replace")
            return _extract_zip_bytes_to_temp_dir(self.decode(), dirname=dirname)

        raise ValueError(f"Unsupported PATH payload mode: {mode!r}")

    @classmethod
    def from_python_object(
        cls,
        obj: object,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object] | None:
        if isinstance(obj, Path):
            return cls.from_value(obj, metadata=metadata, codec=codec)
        return None

    @classmethod
    def from_value(
        cls,
        path: Path,
        *,
        metadata: Mapping[bytes, bytes] | None = None,
        codec: int | None = None,
    ) -> Serialized[object]:
        path = path.expanduser()
        source_os = _current_os_name()

        if path.is_file():
            payload = path.read_bytes()
            merged = _metadata_merge(
                metadata,
                {
                    M_PATH_MODE: PATH_MODE_FILE,
                    M_FILENAME: path.name.encode("utf-8", errors="replace"),
                    M_PATH_OS: source_os,
                },
            )
            return cls.build(
                tag=cls.TAG,
                data=payload,
                metadata=merged,
                codec=codec,
            )

        if path.is_dir():
            try:
                total = _dir_total_bytes(path)
            except Exception:
                total = MAX_INLINE_DIR_BYTES + 1

            if total < MAX_INLINE_DIR_BYTES:
                payload = _zip_directory_to_bytes(path)
                merged = _metadata_merge(
                    metadata,
                    {
                        M_PATH_MODE: PATH_MODE_DIR_ZIP,
                        M_DIRNAME: path.name.encode("utf-8", errors="replace"),
                        M_PATH_OS: source_os,
                    },
                )
                return cls.build(
                    tag=cls.TAG,
                    data=payload,
                    metadata=merged,
                    codec=codec,
                )

        home = Path.home()
        try:
            raw_path = str(path.relative_to(home))
            raw_path = "~" if not raw_path else f"~/{raw_path}"
        except ValueError:
            raw_path = str(path)

        merged = _metadata_merge(
            metadata,
            {
                M_PATH_MODE: PATH_MODE_PATH,
                M_PATH_OS: source_os,
            },
        )
        return cls.build(
            tag=cls.TAG,
            data=raw_path.encode("utf-8"),
            metadata=merged,
            codec=codec,
        )


@dataclass(frozen=True, slots=True)
class IPAddressSerialized(LogicalSerialized[
    ipaddress.IPv4Address
    | ipaddress.IPv6Address
    | ipaddress.IPv4Interface
    | ipaddress.IPv6Interface
    | ipaddress.IPv4Network
    | ipaddress.IPv6Network
]):
    """
    IP payload:
        payload = canonical utf-8 string
        metadata:
            k = 4a | 6a | 4i | 6i | 4n | 6n
    """

    TAG: ClassVar[int] = Tags.IPADDRESS

    @property
    def value(self):
        text = self.decode().decode("utf-8")
        kind = _metadata_bytes(self.metadata, M_KIND)

        if kind == K_IP4_ADDR:
            return ipaddress.IPv4Address(text)
        if kind == K_IP6_ADDR:
            return ipaddress.IPv6Address(text)
        if kind == K_IP4_IFACE:
            return ipaddress.IPv4Interface(text)
        if kind == K_IP6_IFACE:
            return ipaddress.IPv6Interface(text)
        if kind == K_IP4_NET:
            return ipaddress.IPv4Network(text)
        if kind == K_IP6_NET:
            return ipaddress.IPv6Network(text)

        raise ValueError(f"Invalid IPADDRESS metadata kind: {kind!r}")


@dataclass(frozen=True, slots=True)
class URLSerialized(LogicalSerialized[str]):
    """
    URL payload:
        payload = utf-8 URL string
    """

    TAG: ClassVar[int] = Tags.URL

    @property
    def value(self) -> str:
        return self.decode().decode("utf-8")


# ============================================================================
# registration
# ============================================================================

for cls in LogicalSerialized.__subclasses__():
    Tags.register_class(cls, tag=cls.TAG)

for t, cls in (
    (date, DateSerialized),
    (datetime, DatetimeSerialized),
    (time, TimeSerialized),
    (timedelta, TimedeltaSerialized),
    (Decimal, DecimalSerialized),
    (uuid.UUID, UUIDSerialized),
    (complex, ComplexNumberSerialized),
    (bytes, BytesSerialized),
    (bytearray, BytesSerialized),
    (memoryview, BytesSerialized),
    (Path, PathSerialized),
    (URL, URLSerialized),
    (ipaddress.IPv4Address, IPAddressSerialized),
    (ipaddress.IPv6Address, IPAddressSerialized),
    (ipaddress.IPv4Interface, IPAddressSerialized),
    (ipaddress.IPv6Interface, IPAddressSerialized),
    (ipaddress.IPv4Network, IPAddressSerialized),
    (ipaddress.IPv6Network, IPAddressSerialized),
    (timezone, TimezoneSerialized),
):
    Tags.register_class(cls, pytype=t)

if ZoneInfo is not None:
    Tags.register_class(TimezoneSerialized, pytype=ZoneInfo)