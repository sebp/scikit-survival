"""Narwhals-backed concatenation helper.

Pandas-specific concatenation lives in :mod:`sksurv.util` alongside the
public ``safe_concat`` dispatcher. This module is intentionally limited
to Narwhals so the dataframe boundary modules in
``sksurv._dataframe`` stay free of backend-specific imports.
"""

import narwhals.stable.v2 as nw

from ._categorical_semantics import get_semantic_categories

__all__ = ["concat_narwhals_preserving_categories"]


def concat_narwhals_preserving_categories(objs, axis):
    """Concatenate Narwhals-supported dataframes / series along ``axis``.

    Parameters
    ----------
    objs : sequence
        Native frames / series from a single Narwhals-supported backend.
        ``polars.LazyFrame`` inputs are collected before concatenation.
    axis : {0, 1}
        Already-normalised concatenation axis. ``0`` stacks rows,
        ``1`` stacks columns.
    """
    frames = []
    for obj in objs:
        nw_obj = nw.from_native(obj, allow_series=True)
        if isinstance(nw_obj, nw.LazyFrame):
            nw_obj = nw_obj.collect()
        if isinstance(nw_obj, nw.Series):
            nw_obj = nw_obj.to_frame()
        frames.append(nw_obj)

    categories = {}
    for nw_df in frames:
        for col_name, dtype in nw_df.schema.items():
            if not isinstance(dtype, (nw.Categorical, nw.Enum)):
                continue
            col_cats = get_semantic_categories(nw_df.get_column(col_name))
            if col_name in categories:
                if axis == 1:
                    raise ValueError(f"duplicate columns {col_name}")
                if categories[col_name] != col_cats:
                    raise ValueError(f"categories for column {col_name} do not match")
            else:
                categories[col_name] = col_cats

    how = "horizontal" if axis == 1 else "vertical"
    return nw.concat(frames, how=how).to_native()
