# yggdrasil.concurrent

Bounded job execution utilities for large or unbounded job streams.

## Main exports

```python
from yggdrasil.concurrent import Job, JobPoolExecutor
```

## Example using `as_completed`

```python
from yggdrasil.concurrent import Job, JobPoolExecutor

jobs = [Job.make(lambda x=x: x * x) for x in range(5)]

with JobPoolExecutor(max_workers=2, max_in_flight=4) as pool:
    for result in pool.as_completed(jobs):
        print(result.value)
```
