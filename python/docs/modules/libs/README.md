# yggdrasil.libs

`yggdrasil.libs` documents optional dependency and extension patterns used across Yggdrasil data modules.

Use it when your code may run in environments where some engines are optional (Spark, Polars, Pandas).

---

## What this module area covers

- Dependency guard functions for optional libraries.
- Conversion helpers tied to dataframe/type interoperability.
- Integration support for Spark/Polars-heavy projects.

---

## Bootstrap: dependency guard pattern

```python
# Example pattern when optional dependencies are expected
# from yggdrasil.libs import require_polars
# require_polars()
```

The idea is to fail fast with informative errors when a runtime lacks required libraries.

---

## Bootstrap: optional engine branching

```python
def process_frame(frame, engine: str):
    if engine == "spark":
        # spark-specific path
        return frame
    if engine == "polars":
        # polars-specific path
        return frame
    raise ValueError(f"Unsupported engine: {engine}")
```

---

## Related submodule

- [libs.extensions](extensions/README.md): dataframe extension helpers.
