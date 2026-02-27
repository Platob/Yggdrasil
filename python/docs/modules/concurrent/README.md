# yggdrasil.concurrent

`JobPoolExecutor` and `Job` for streaming large or infinite workloads through a bounded thread pool.

Use this when you need to fan-out over a huge / infinite job stream without exhausting memory, controlling at most `max_in_flight` futures in flight at once.

## Key exports

```python
from yggdrasil.concurrent import JobPoolExecutor, Job
```

---

## `Job` — immutable unit of work

```python
Job.make(func, *args, **kwargs) -> Job
job.run()              # execute synchronously
job.fire_and_forget()  # launch in background thread, no result
```

---

## `JobPoolExecutor` — bounded thread pool

```python
JobPoolExecutor(
    max_workers=None,        # thread pool size (default: CPU count)
    max_in_flight=None,      # max concurrent in-flight futures (default: max_workers × 2)
    job_name_prefix="",      # thread name prefix
)
```

---

## Bootstrap: simple job pool

```python
from yggdrasil.concurrent import JobPoolExecutor, Job

def process(item: str) -> dict:
    return {"item": item, "n": len(item)}

jobs = (Job.make(process, item) for item in ["a", "bb", "ccc"])

with JobPoolExecutor(max_workers=4, max_in_flight=8) as pool:
    for future in pool.as_completed(jobs, ordered=False):
        print(future.result())
```

---

## Bootstrap: ordered results (submission order)

```python
from yggdrasil.concurrent import JobPoolExecutor, Job

def fetch(url: str) -> bytes:
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return r.read()

urls = ["https://example.com/a", "https://example.com/b"]
jobs = (Job.make(fetch, url) for url in urls)

with JobPoolExecutor(max_workers=4) as pool:
    for future in pool.as_completed(jobs, ordered=True):
        print(len(future.result()), "bytes")
```

---

## Bootstrap: infinite generator with back-pressure

```python
from yggdrasil.concurrent import JobPoolExecutor, Job

def do_work(n: int) -> int:
    return n * n

def job_stream():
    for i in range(10_000_000):   # huge / infinite
        yield Job.make(do_work, i)

with JobPoolExecutor(max_workers=16, max_in_flight=64) as pool:
    for future in pool.as_completed(job_stream(), ordered=False):
        result = future.result()
        # pool submits new jobs only when in-flight count drops below max_in_flight
```

---

## Bootstrap: fire-and-forget background tasks

```python
from yggdrasil.concurrent import Job

def send_alert(msg: str) -> None:
    import requests
    requests.post("https://hooks.example.com/alert", json={"text": msg}, timeout=5)

# non-blocking — does not return a future
Job.make(send_alert, "pipeline completed").fire_and_forget()
```

---

## `as_completed` options

```python
pool.as_completed(
    job_generator,    # Iterable[Job]
    ordered=False,    # False: completion order | True: submission order
)
# yields: concurrent.futures.Future
```
