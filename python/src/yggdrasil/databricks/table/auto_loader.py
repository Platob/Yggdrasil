"""On-cluster Auto Loader (``cloudFiles``) ingestion entry point.

:func:`auto_load` is what the Databricks job built by
:meth:`yggdrasil.databricks.table.table.Table.auto_loader` actually runs on
the cluster (imported by ``ygg-run`` from the shipped ygg wheel). It streams
files from a source path into a Unity Catalog table via Spark Structured
Streaming + Databricks Auto Loader — incremental, exactly-once, schema-evolving
— so a table keeps absorbing new files dropped at *source* without a bespoke
pipeline.

Kept as a plain module-level function (typed params) so the runner can both
import it (``module:qualname``) and coerce the positional string job parameters
to the signature.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["auto_load"]


def auto_load(
    table: str,
    source: str,
    file_format: str = "parquet",
    checkpoint: str = "",
    available_now: bool = True,
    clean_source: bool = False,
    clean_source_retention: str = "8 days",
) -> dict[str, Any]:
    """Ingest files under *source* into *table* with Databricks Auto Loader.

    Each micro-batch is **cast to the target table's schema before the append**
    (via ``foreachBatch``): the target's columns are projected in order, source
    columns cast to the target type and missing ones NULL-filled, so the write is
    schema-stable and tolerant of source drift — extra columns are dropped and
    type mismatches are cast rather than failing or evolving the target schema.
    Appends are made idempotent with Delta's ``(txnAppId, txnVersion)`` marker
    keyed on the batch id, preserving exactly-once across task retries.

    Args:
        table: Target table, ``catalog.schema.table``.
        source: Cloud path Auto Loader watches (``s3://…`` / ``/Volumes/…``).
        file_format: ``cloudFiles.format`` — ``parquet`` / ``json`` / ``csv`` /
            ``avro`` / ``delta`` …
        checkpoint: Streaming checkpoint + schema location. Empty → derived as
            ``<table-storage-location>/_ygg_autoloader`` from ``DESCRIBE
            DETAIL`` so each table gets its own, co-located with its data.
        available_now: ``True`` (default) runs a single
            ``Trigger.AvailableNow`` micro-batch sweep of everything currently
            at *source* then stops — the shape a scheduled / file-arrival job
            wants. ``False`` runs a continuous 1-minute micro-batch stream.
        clean_source: ``True`` makes Auto Loader **delete each source file once
            it's been ingested and is older than the retention window**
            (``cloudFiles.cleanSource = DELETE``) so a staging area doesn't grow
            without bound. Cleanup runs at the start of a later micro-batch — it
            does **not** delete files within the same one-shot ``AvailableNow``
            sweep that ingests them — so it's a rolling janitor, not immediate.
            Default ``False`` leaves processed files in place.
        clean_source_retention: Retention window for *clean_source*
            (``cloudFiles.cleanSource.retentionDuration``). Databricks requires
            an interval **greater than 7 days**; default ``"8 days"``.

    Returns a small summary dict (table + resolved checkpoint + rows ingested
    when known) — handy in the job run output / when called locally.
    """
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    if not checkpoint:
        detail = spark.sql(f"DESCRIBE DETAIL {table}").collect()[0]
        location = (detail["location"] or "").rstrip("/")
        if not location:
            raise ValueError(
                f"auto_load: could not resolve a storage location for {table!r} "
                f"to derive a checkpoint; pass `checkpoint=` explicitly."
            )
        checkpoint = f"{location}/_ygg_autoloader"

    logger.info(
        "[YGG][AUTOLOADER] %s ← %s (format=%s, checkpoint=%s, available_now=%s)",
        table, source, file_format, checkpoint, available_now,
    )

    reader = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", file_format)
        # Persist the inferred schema next to the checkpoint and let new
        # columns flow in rather than failing the stream.
        .option("cloudFiles.schemaLocation", f"{checkpoint}/_schema")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
    )
    if clean_source:
        # Self-cleaning staging: delete each file once ingested + committed and
        # older than the retention window (Databricks requires > 7 days).
        reader = (
            reader
            .option("cloudFiles.cleanSource", "DELETE")
            .option("cloudFiles.cleanSource.retentionDuration", clean_source_retention)
        )
    frame = reader.load(source)

    # Idempotent app id so a re-run of the same micro-batch (e.g. after a task
    # retry) is deduplicated by Delta's (txnAppId, txnVersion) transaction marker
    # — preserving the exactly-once ``.toTable`` gave before foreachBatch.
    app_id = f"ygg_autoloader::{table}"

    def _write_batch(batch_df: Any, batch_id: int) -> None:
        # Cast/align each micro-batch to the **target table** schema before the
        # append: project the target's columns in order, casting each source
        # column to the target type and NULL-filling any the batch doesn't carry.
        # Per-batch alignment keeps the write schema-stable (no ``mergeSchema``)
        # and tolerant of source drift — extra source columns are dropped and
        # type mismatches are cast, rather than failing or silently evolving the
        # target table's schema.
        from pyspark.sql import functions as F

        target_schema = batch_df.sparkSession.table(table).schema
        present = set(batch_df.columns)
        projection = [
            (F.col(field.name) if field.name in present else F.lit(None))
            .cast(field.dataType).alias(field.name)
            for field in target_schema.fields
        ]
        (
            batch_df.select(*projection)
            .write.format("delta").mode("append")
            # Idempotent append: Delta no-ops a replay of the same batch id.
            .option("txnAppId", app_id)
            .option("txnVersion", batch_id)
            .saveAsTable(table)
        )

    writer = (
        frame.writeStream.option("checkpointLocation", checkpoint)
        .foreachBatch(_write_batch)
    )
    writer = (
        writer.trigger(availableNow=True)
        if available_now
        else writer.trigger(processingTime="1 minute")
    )
    query = writer.start()
    query.awaitTermination()

    summary: dict[str, Any] = {"table": table, "source": source, "checkpoint": checkpoint}
    try:
        progress = query.lastProgress
        if progress:
            summary["rows"] = int(progress.get("numInputRows", 0))
    except Exception:  # noqa: BLE001 — progress is best-effort telemetry
        pass
    logger.info("[YGG][AUTOLOADER] into %s finished: %s", table, summary)
    return summary
