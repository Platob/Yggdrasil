import polars as pl
import datetime as dt

class DataFrame(pl.DataFrame):
    def check_datetime_fields(
        self,
        time_zone: str | None = None,
        naive_time_zone: str | None = None,
    ) -> pl.DataFrame:
        expressions = []

        for name, dtype in self.schema.items():
            if isinstance(dtype, pl.Datetime):
                expression = pl.col(name)
                append = False

                column_time_zone = getattr(dtype, "time_zone", None)

                if naive_time_zone and not column_time_zone:
                    expression = expression.dt.replace_time_zone(naive_time_zone)
                    append, column_time_zone = True, naive_time_zone

                if time_zone and column_time_zone != time_zone:
                    expression = expression.dt.convert_time_zone(time_zone)
                    append, column_time_zone = True, time_zone

                if append:
                    expressions.append(expression.alias(name))

        return self.with_columns(expressions) if expressions else self


__all__ = [
    "DataFrame"
]