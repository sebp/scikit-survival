from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype


def to_polars_dataframe(df):
    """Convert pandas test data to polars without using ``__dataframe__``.

    The explicit column-wise conversion preserves pandas categorical category
    declarations as ``pl.Enum`` and avoids polars' deprecated dataframe
    interchange protocol path.
    """
    import polars as pl

    columns = {}
    for name, col in df.items():
        values = col.astype(object).where(pd.notna(col), None).tolist()
        if isinstance(col.dtype, CategoricalDtype) and all(isinstance(cat, str) for cat in col.cat.categories):
            columns[name] = pl.Series(name, values, dtype=pl.Enum(col.cat.categories.tolist()))
        else:
            columns[name] = values
    return pl.DataFrame(columns)


def expected_one_hot_data(data):
    expected = []
    for nam, col in data.items():
        if hasattr(col, "cat"):
            for cat in col.cat.categories[1:]:
                name = f"{nam}={cat}"
                s = pd.Series(col == cat, dtype=np.float64)
                expected.append((name, s))
        else:
            expected.append((nam, col))

    return pd.DataFrame.from_dict(OrderedDict(expected))


def make_one_hot_categorical_data(n_samples=117):
    rnd = np.random.default_rng(51365192)
    data_num = pd.DataFrame(rnd.random((n_samples, 5)), columns=[f"N{i}" for i in range(5)])

    dat_cat = pd.DataFrame(
        OrderedDict(
            [
                ("binary_1", pd.Categorical.from_codes(rnd.binomial(1, 0.6, n_samples), ["Yes", "No"])),
                ("binary_2", pd.Categorical.from_codes(rnd.binomial(1, 0.376, n_samples), ["East", "West"])),
                ("trinary", pd.Categorical.from_codes(rnd.binomial(2, 0.76, n_samples), ["Green", "Blue", "Red"])),
                (
                    "many",
                    pd.Categorical.from_codes(
                        rnd.binomial(5, 0.47, n_samples), ["One", "Two", "Three", "Four", "Five", "Six"]
                    ),
                ),
            ]
        )
    )
    data = pd.concat((data_num, dat_cat), axis=1)
    return data, expected_one_hot_data(data)


def make_clinical_kernel_expected(with_ordinal=True, with_nominal=True, with_continuous=True):
    mat_age = np.array(
        [
            [1.0, 0.9625, 0.925, 0.575, 0.0],
            [0.9625, 1.0, 0.9625, 0.6125, 0.0375],
            [0.925, 0.9625, 1.0, 0.6500, 0.075],
            [0.575, 0.6125, 0.6500, 1.0, 0.425],
            [0.0, 0.0375, 0.075, 0.425, 1.0],
        ]
    )
    mat_node_size = np.array(
        [
            [1.0, 2 / 3, 2 / 3, 1 / 3, 2 / 3],
            [2 / 3, 1.0, 1 / 3, 0.0, 1.0],
            [2 / 3, 1 / 3, 1.0, 2 / 3, 1 / 3],
            [1 / 3, 0.0, 2 / 3, 1.0, 0.0],
            [2 / 3, 1.0, 1 / 3, 0.0, 1.0],
        ]
    )
    mat_node_spread = np.array(
        [
            [1.0, 0.0, 1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5, 1.0],
            [1.0, 0.0, 1.0, 0.5, 0.0],
            [0.5, 0.5, 0.5, 1.0, 0.5],
            [0.0, 1.0, 0.0, 0.5, 1.0],
        ]
    )
    mat_metastasis = np.array(
        [
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
            [1, 0, 1, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1],
        ],
        dtype=float,
    )

    included = []
    if with_continuous:
        included.append(mat_age)
    if with_ordinal:
        included.append(mat_node_size)
        included.append(mat_node_spread)
    if with_nominal:
        included.append(mat_metastasis)

    expected = included[0]
    for i in range(1, len(included)):
        expected += included[i]
    expected /= len(included)
    return expected


def make_clinical_kernel_pandas_data(with_ordinal=True, with_nominal=True, with_continuous=True):
    data = {
        "age": [20, 23, 26, 54, 100],
        "lymph node size": [2, 1, 3, 4, 1],
        "lymph node spread": ["distant", "none", "distant", "close", "none"],
        "metastasis": ["yes", "no", "yes", "yes", "no"],
    }
    data_s = {}
    if with_continuous:
        data_s["age"] = data["age"]
    if with_ordinal:
        data_s["lymph node size"] = pd.Categorical(data["lymph node size"], categories=[1, 2, 3, 4], ordered=True)
        data_s["lymph node spread"] = pd.Categorical(
            data["lymph node spread"], categories=["none", "close", "distant"], ordered=True
        )
    if with_nominal:
        data_s["metastasis"] = pd.Categorical(data["metastasis"], categories=["no", "yes"], ordered=False)
    expected = make_clinical_kernel_expected(
        with_ordinal=with_ordinal, with_nominal=with_nominal, with_continuous=with_continuous
    )
    return pd.DataFrame(data_s), expected
