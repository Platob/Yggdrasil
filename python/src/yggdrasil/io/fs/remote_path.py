"""Abstract base for network-backed :class:`Path` implementations.

Why this exists
---------------
The abstract :class:`Path` is already remote-friendly by default —
its transaction-buffer machinery downloads on first read, splices
positional writes into a local :class:`BytesIO`, and commits via a
single ``_pwrite`` on flush. ``LocalPath`` is the special case that
overrides that with a long-lived file descriptor.

What was missing was a taxonomic seam: every remote backend (S3,
Databricks, future Azure / GCS / SFTP / WebDAV) had to repeat the
same handful of class-level facts —

- ``is_local`` is hard-False.
- ``fileno()`` raises (no kernel fd to expose).
- ``open_mmap`` returns ``None`` (memory-mapping a remote object is
  meaningless).

…and there was no single ``isinstance(p, RemotePath)`` check
callers could use to branch on remote-vs-local without enumerating
every concrete backend.

:class:`RemotePath` is that seam. It does not introduce new I/O
contracts on top of :class:`Path` — concrete subclasses still
implement the same seven hooks (:meth:`full_path`, :meth:`_stat`,
:meth:`_ls`, :meth:`_mkdir`, :meth:`_remove_file`, :meth:`_remove_dir`,
:meth:`_pread`, :meth:`_pwrite`). It just pins down the always-true
"this is not a local file" answers and gives the codebase a stable
type to dispatch on.
"""

from __future__ import annotations

from abc import ABC

from yggdrasil.io.fs.path import Path

__all__ = ["RemotePath"]


class RemotePath(Path, ABC):
    """Abstract :class:`Path` for network-backed backends.

    Concrete subclasses inherit the base :class:`Path`'s
    transaction-buffer machinery by default — they only override
    :meth:`pread` / :meth:`pwrite` / :meth:`write_stream` when the
    backend has a cheaper primitive than download-and-slice or
    upload-the-whole-buffer (e.g. S3 Range GETs, DBFS FUSE).
    """

    __slots__ = ()

    @property
    def is_local(self) -> bool:
        return False
