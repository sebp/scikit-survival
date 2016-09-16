# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import datetime
from os.path import join

import pandas
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import ParameterGrid

from survival.svm.survival_svm import FastSurvivalSVM
from survival.svm.naive_survival_svm import NaiveSurvivalSVM
from survival.io import loadarff
from survival import column


DATASETS = {'veteran': {
                 'filename': 'veteran.arff',
                 'label': ['Survival_in_days', 'Status'],
                 'outcome': "'dead'"
            },
            'whas500': {
                 'filename': 'whas500.arff',
                 'label': ['lenfol', 'fstat'],
                 'outcome': "1"
            },
            'actg320_aids': {
                 'filename': 'actg320.arff',
                 'label': ['time', 'censor', 'time_d', 'censor_d'],
                 'outcome': "1"
            },
            'actg320_death': {
                 'filename': 'actg320.arff',
                 'label': ['time_d', 'censor_d', 'time', 'censor'],
                 'outcome': "1"
            },
            'breast-cancer': {
                'filename': 'breast_cancer_GSE7390-metastasis.arff',
                'label': ["t.tdm", "e.tdm"],
                'outcome': "'1'"}
           }


def load_dataset(name, base_dir):
    meta = DATASETS[name]

    data = loadarff(join(base_dir, meta['filename']))
    x_orig = data.drop(meta['label'], axis=1)
    x = column.categorical_to_numeric(column.standardize(x_orig))

    y_time = data.loc[:, meta['label'][0]]
    y_event = data.loc[:, meta['label'][1]] == meta['outcome']
    y = numpy.empty(dtype=[('event', bool), ('time', float)], shape=x.shape[0])
    y['event'] = y_event.values
    y['time'] = numpy.log(y_time.values)

    assert len(y_event.value_counts()) == 2

    return x.values, y


def get_params_grid(method):
    if method in ('l2_ranking_regression', 'l2_ranking_regression_kernel'):
        param_grid = {'alpha': 2. ** numpy.arange(-12, 13, 2),
                      'rank_ratio': numpy.arange(0, 1 + 1e-8, 0.05)}
    else:
        param_grid = {'alpha': 2. ** numpy.arange(-12, 13, 2)}

    return param_grid


def get_estimator(method):
    if method == 'l2_ranking_regression':
        estimator = FastSurvivalSVM(optimizer='rbtree', random_state=0,
                                    fit_intercept=True, max_iter=1000, tol=1e-6)
    elif method == 'l2_ranking':
        estimator = FastSurvivalSVM(optimizer='rbtree', random_state=0,
                                    rank_ratio=1.0, fit_intercept=False, max_iter=1000, tol=1e-6)
    elif method == 'l2_regression':
        estimator = FastSurvivalSVM(optimizer='rbtree', random_state=0,
                                    rank_ratio=0.0, fit_intercept=True, max_iter=1000, tol=1e-6)
    elif method == 'l1':
        estimator = NaiveSurvivalSVM(loss='hinge', random_state=0, dual=True, max_iter=1000, tol=1e-6)
    else:
        raise ValueError('unknown method: %s' % method)

    return estimator


def train_test_model(data):
    train_index, test_index, params, fold = data

    # Training
    est = clone(estimator)
    est.set_params(**params)

    ret = params.copy()
    try:
        est.fit(x[train_index, :], y[train_index])

        # Testing
        p = est.predict(x[test_index, :])
        test_y = y[test_index]
        c = concordance_index_censored(test_y['event'], test_y['time'], p)

        ret['c-index'] = c[0]
        # for c-index, the sign of the predictions is flipped, flip it again for regression
        p_regression = -p[test_y['event']]

        # convert from log-scale back to original scale and compute RMSE
        ret['error'] = numpy.sqrt(mean_squared_error(numpy.exp(test_y['time'][test_y['event']]),
                                                     numpy.exp(p_regression)))
        ret['n_events'] = numpy.sum(test_y['event'])
    except Exception as e:
        # log errors to IPython profile's log files
        Application.instance().log.exception(e)
        ret['c-index'] = float('nan')
        ret['error'] = float('nan')
        ret['n_events'] = float('nan')

    ret['fold'] = fold

    return ret


def params_cv(param_grid, train_test_iter):
    params_iter = ParameterGrid(param_grid)
    for params in params_iter:
        for i, (train, test) in enumerate(train_test_iter):
            yield train, test, params, i


if __name__ == '__main__':
    from ipyparallel import Client
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Perform grid search on specified dataset using specified method.")
    parser.add_argument('-m', '--method',
                        choices=['l1', 'l2_ranking', 'l2_regression', 'l2_ranking_regression'],
                        required=True, help="Name of method to evaluate")
    parser.add_argument('-d', '--dataset',
                        choices=list(DATASETS.keys()),
                        required=True, help="Name of dataset to load")
    parser.add_argument('--base-dir', default="data", help="Path to directory containing datasets")
    parser.add_argument('-p', '--profile', default='default', help="Name of IPython profile to use")

    args = parser.parse_args()

    rc = Client(profile=args.profile)
    dview = rc[:]
    lview = rc.load_balanced_view()

    with dview.sync_imports():
        from traitlets.config import Application
        from survival.metrics import concordance_index_censored
        from sklearn.base import clone
        from sklearn.metrics import mean_squared_error
        import numpy

    _x, _y = load_dataset(args.dataset, args.base_dir)
    _estimator = get_estimator(args.method)
    _param_grid = get_params_grid(args.method)
    print(_x.shape)

    # distribute data to engines
    dview.push({'x': _x, 'y': _y, 'estimator': _estimator})

    _train_test_iter = ShuffleSplit(_x.shape[0], n_iter=200, train_size=0.5, random_state=0)
    _params_iter = params_cv(_param_grid, _train_test_iter)

    _results = lview.map(train_test_model, _params_iter, block=True, chunksize=10)

    _data = pandas.DataFrame(_results)
    _output = 'results-%s-%s.csv' % (args.dataset, args.method)

    with open(_output, 'w') as fp:
        dt = datetime.datetime.now()
        fp.write("# Created on {0}\n".format(dt.strftime("%Y-%m-%d %H:%M")))
        fp.write("# {0}\n".format(" ".join(sys.argv)))
        _data.to_csv(fp)
