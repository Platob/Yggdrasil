from __future__ import annotations

from mongoengine import Document


class HttpDocument(Document):
    """Common base class for models using the HTTP Mongo gateway."""

    meta = {"abstract": True}
