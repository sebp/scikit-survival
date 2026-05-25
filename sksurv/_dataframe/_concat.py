"""Dataframe-aware concatenation helpers."""

import narwhals.stable.v2 as nw
import pandas as pd
from pandas.api.types import CategoricalDtype

from ._categorical_semantics import get_semantic_categories
from ._input import SUPPORTED_DATAFRAME_INPUT_TYPES, get_dataframe_library

__all__ = ["concat_dataframes_preserving_categories"]


def concat_dataframes_preserving_categories(objs, *args, **kwargs):
    axis = _normalize_axis(kwargs.pop("axis", 0))
    objs = list(objs)
    if not objs:
        raise ValueError("No objects to concatenate")

    first = objs[0]
    if _is_pandas_concat_input(first):
        for obj in objs[1:]:
            if not _is_pandas_concat_input(obj):
                raise TypeError(f"mixed backends; all inputs must be pandas, but found {type(obj)!r}")
        return _concat_pandas_preserving_categories(objs, *args, axis=axis, **kwargs)

    library = get_dataframe_library(first, allow_series=True)
    if library is not None:
        for obj in objs[1:]:
            obj_library = get_dataframe_library(obj, allow_series=True)
            if obj_library is None or obj_library.name != library.name:
                raise TypeError(f"mixed backends; all inputs must be {library.name}, but found {type(obj)!r}")
        if args or kwargs:
            unsupported = list(kwargs.keys())
            if args:
                unsupported = ["<positional>"] * len(args) + unsupported
            raise TypeError(
                f"safe_concat {library.name} path does not accept {unsupported!r}; "
                f"only ``axis`` is honoured for {library.name} input"
            )
        return _concat_narwhals_preserving_categories(objs, axis=axis)

    raise TypeError(f"safe_concat: unsupported input type {type(first)!r}; expected {SUPPORTED_DATAFRAME_INPUT_TYPES}")


def _normalize_axis(axis):
    if axis == "index":
        return 0
    if axis == "columns":
        return 1
    if axis in (0, 1):
        return axis
    raise ValueError(f"axis must be 0 or 1, got {axis!r}")


def _is_pandas_concat_input(obj):
    return nw.dependencies.is_pandas_dataframe(obj) or nw.dependencies.is_pandas_series(obj)


def _concat_pandas_preserving_categories(objs, *args, axis=0, **kwargs):
    categories = {}
    for df in objs:
        if isinstance(df, pd.Series):
            if isinstance(df.dtype, CategoricalDtype):
                categories[df.name] = {"categories": df.cat.categories, "ordered": df.cat.ordered}
        else:
            dfc = df.select_dtypes(include=["category"])
            new_dtypes = {}
            for name, s in dfc.items():
                if name in categories:
                    if axis == 1:
                        raise ValueError(f"duplicate columns {name}")
                    if not categories[name]["categories"].equals(s.cat.categories):
                        raise ValueError(f"categories for column {name} do not match")
                else:
                    categories[name] = {"categories": s.cat.categories, "ordered": s.cat.ordered}
                new_dtypes[name] = "str"
            df = df.astype(new_dtypes)

    concatenated = pd.concat(objs, *args, axis=axis, **kwargs)
    concatenated = concatenated.astype({name: pd.CategoricalDtype(**params) for name, params in categories.items()})
    return concatenated


def _concat_narwhals_preserving_categories(objs, axis):
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
