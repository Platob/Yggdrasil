from .lib import mongoengine

__all__ = [
    "connect"
]

def connect(
    db: str = None,
    alias: str = mongoengine.DEFAULT_CONNECTION_NAME,
    host: str | None = None,
    **kwargs
):
    return mongoengine.connect(
        db=db,
        alias=alias,
        host=host,
        **kwargs
    )