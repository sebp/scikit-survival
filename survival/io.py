import numpy
from scipy.io.arff import loadarff as scipy_loadarff
import pandas

__all__ = ["loadarff"]


def _to_pandas(data, meta):
    data_dict = {}
    attrnames = meta.names()
    for i, name in enumerate(attrnames):
        tp, attr_format = meta[name]
        if tp == "nominal":
            raw = []
            for j, b in enumerate(data[name]):
                # replace missing values with NaN
                if b == b'?':
                    raw.append(numpy.nan)
                else:
                    raw.append(b.decode())

            data_dict[name] = pandas.Categorical(raw, categories=attr_format, ordered=False)
        else:
            arr = data[name]
            p = pandas.Series(arr, dtype=arr.dtype)
            data_dict[name] = p

    # currently, this step converts all pandas.Categorial columns back to pandas.Series
    return pandas.DataFrame(data_dict)


def loadarff(filename):
    """Load ARFF file

    Parameters
    ----------
    filename : string
        Path to ARFF file

    Returns
    -------
    data_frame : :class:`pandas.DataFrame`
        DataFrame containing data of ARFF file
    """
    data, meta = scipy_loadarff(filename)
    return _to_pandas(data, meta)
