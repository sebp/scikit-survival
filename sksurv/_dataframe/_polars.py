"""Polars input predicates for the dataframe boundary.

This module only decides whether a native object is a polars frame or
series. Dataframe-library-neutral normalization and processing belong in
``_input.py``, ``_categorical_semantics.py``, or ``_categorical_encoding.py``
after values enter Narwhals.
"""

import narwhals.stable.v2 as nw

__all__ = ["LIBRARY", "PolarsDataFrameLibrary"]


class PolarsDataFrameLibrary:
    name = "polars"
    # Each native dataframe / series type is listed separately so that the
    # routing layer can build a single Oxford-formatted "supported input
    # types" string across libraries.
    dataframe_display_names = ("polars.DataFrame",)
    series_display_names = ("polars.Series",)
    dataframe_display_name = " or ".join(dataframe_display_names)
    series_display_name = " or ".join(series_display_names)

    @staticmethod
    def is_dataframe(obj):
        return nw.dependencies.is_polars_dataframe(obj)

    @staticmethod
    def is_series(obj):
        return nw.dependencies.is_polars_series(obj)

    @staticmethod
    def is_non_numeric_cast_error(exc):
        """Return True iff ``exc`` was raised by polars rejecting a
        non-numeric string ``cast`` (e.g. ``"foo".cast(Int64)``).

        Implemented as a class-name + module-name check instead of an
        ``isinstance`` so this module doesn't import ``polars`` eagerly
        and stays importable when polars isn't installed.
        """
        return exc.__class__.__name__ == "InvalidOperationError" and exc.__class__.__module__.startswith("polars.")


LIBRARY = PolarsDataFrameLibrary()
