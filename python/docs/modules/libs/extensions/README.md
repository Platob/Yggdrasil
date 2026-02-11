# yggdrasil.libs.extensions

Extension helpers for dataframe workflows that benefit from concise utility operations.

This area is useful when you want shared reusable logic for joins, sampling/resampling, and dataframe convenience operations.

---

## Typical usage

- Repeated join-with-merge-column patterns
- Time-window resampling tasks
- Small helper APIs for dataframe interoperability

---

## Bootstrap: extension-style helper call

```python
# Example shape for extension helper usage
# from yggdrasil.libs.extensions import join_coalesced
# output = join_coalesced(left_df, right_df, on="id")
```

---

## Bootstrap: compose with a pipeline stage

```python
def transform_features(df):
    # apply extension helpers in deterministic order
    # df = join_coalesced(df, ref_df, on="entity_id")
    return df
```

---

## Recommendations

- Keep extension operations deterministic and side-effect free.
- Document key assumptions (sort order, null semantics, dedup strategy).
- Add tests for overlapping-column and missing-key edge cases.
