# yggdrasil.pyutils

General-purpose helpers, especially retry and parallel decorators.

## Retry decorator

```python
from yggdrasil.pyutils import retry

@retry(tries=3, delay=0.2, backoff=2)
def read_remote() -> str:
    return "ok"
```

## Parallelize decorator

```python
from yggdrasil.pyutils import parallelize

@parallelize(max_workers=4)
def square(x: int) -> int:
    return x * x

print(list(square(range(6))))
```
