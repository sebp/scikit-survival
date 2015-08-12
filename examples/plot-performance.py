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
from os.path import basename
import re

import matplotlib as mpl
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

seaborn.set(style='whitegrid')
mpl.rc('legend', fontsize='medium')

METHODS = {'l1': 'Ranking (hinge)',
           'l2_ranking': 'Ranking (squared hinge)',
           'l2_ranking_kernel': 'Ranking (squared hinge, kernel)',
           'l2_ranking_regression': 'Hybrid',
           'l2_ranking_regression_kernel': 'Hybrid (kernel)',
           'l2_regression': 'Regression',
           'coxph': 'Cox PH (ridge)'}

DATASETS = {'actg320_aids': 'AIDS study',
            'actg320_death': 'AIDS study (death)',
            'breast-cancer': 'Breast cancer',
            'veteran': "Veteran's lung cancer",
            'whas500': 'Worcester study'}


def format_data(data):
    params = data.columns - ['c-index', 'error', 'fold', 'n_events']

    if 'rank_ratio' in params:
        mask = (data.loc[:, 'rank_ratio'] > 0) & (data.loc[:, 'rank_ratio'] < 1)
        data = data.loc[mask, :]

    # group by hyper-parameter configurations
    group = data.drop('fold', axis=1).groupby(params.tolist())

    # mean performance per configuration
    group_mean = group.mean()
    # parameters with best mean performance
    idx_max_cindex = group_mean.loc[:, 'c-index'].idxmax()
    idx_min_error = group_mean.loc[:, 'error'].idxmin()

    values_cindex = {}
    values_error = {}
    if len(group_mean.index.names) == 1:
        nam = group_mean.index.names[0]
        values_cindex[nam] = idx_max_cindex
        values_error[nam] = idx_min_error
    else:
        for i, nam in enumerate(group_mean.index.names):
            values_cindex[nam] = idx_max_cindex[i]
            values_error[nam] = idx_min_error[i]

    values_cindex['mean'] = group_mean.loc[idx_max_cindex, 'c-index']
    values_error['mean'] = group_mean.loc[idx_min_error, 'error']

    df = pandas.DataFrame({'best c-index': values_cindex,
                           'best error': values_error}).T

    cindex_dist = data.set_index(params.tolist()).loc[idx_max_cindex, 'c-index']
    rmse_dist = data.set_index(params.tolist()).loc[idx_min_error, 'error']

    return df, cindex_dist, rmse_dist


def _boxplot(methods, *args, **kwargs):
    # Methods are columns and performances are rows
    x = pandas.concat(args, axis=1).T
    x.columns = methods
    return seaborn.boxplot(x, **kwargs)


def add_legend(g, methods):
    legend_data = []
    ax1 = g.axes.flat[-1]

    k = 0
    for c in ax1.get_children():
        # boxes are PathPatch elements
        if isinstance(c, PathPatch):
            key = methods[k]
            legend_data.append((key, c))
            k += 1

    # g.add_legend(legend_data=dict(legend_data), title='Method')
    figlegend = ax1.legend([v[1] for v in legend_data],
                           [v[0] for v in legend_data],
                           loc="upper left", bbox_to_anchor=(1.05, 1),
                           title='Method')
    return figlegend


def plot_cindex():
    if dfa["Data"].nunique() > 1:
        g = seaborn.FacetGrid(dfa, col='Data', ylim=(0, 1), col_wrap=4, aspect=.9, legend_out=True)

        g.map(_boxplot, 'Method', *data_cols, color='Spectral')

        g.set_ylabels('concordance index')
        g.set_xlabels('')

        yticks = numpy.arange(0.1, 1, 0.1)
        for ax in g._left_axes:
            ax.set_yticks(yticks)

        # remove xticks
        for ax in g._bottom_axes:
            ax.get_xaxis().set_major_locator(plt.NullLocator())

        add_legend(g, dfa['Method'].unique())
    else:
        x = pandas.DataFrame(dfa.drop(["Data", "Method"], axis=1).values.T, columns=dfa["Method"])
        g = seaborn.boxplot(x, color='Spectral')
        g.set_title(dfa.loc[0, "Data"])
        g.set_ylabel('concordance index')

    return g


def plot_rmse():
    if dfa["Data"].nunique() > 1:
        g = seaborn.FacetGrid(dfa, col='Data', sharey=False, col_wrap=4, aspect=.9, legend_out=True)

        g.map(_boxplot, 'Method', *data_cols, color=['#c0dbab', '#61a0a0'])
        g.set_ylabels('root mean squared error')
        g.set_xlabels('')

        # remove xticks
        for ax in g._bottom_axes:
            ax.get_xaxis().set_major_locator(plt.NullLocator())

        add_legend(g, dfa['Method'].unique())
    else:
        x = pandas.DataFrame(dfa.drop(["Data", "Method"], axis=1).values.T, columns=dfa["Method"])
        g = seaborn.boxplot(x, color=['#c0dbab', '#61a0a0'])
        g.set_title(dfa.loc[0, "Data"])
        g.set_ylabel('root mean squared error')

    return g


if __name__ == '__main__':
    import argparse

    method_pat = re.compile(
        'results-(.+)-(coxph|l1|l2_regression|l2_ranking_kernel'
        '|l2_ranking_regression_kernel|l2_ranking_regression|l2_ranking)')

    parser = argparse.ArgumentParser(description="Visualize cross-validation results")
    parser.add_argument("-o", "--output", help="Path where plot should be written to")
    parser.add_argument("files", nargs="+", help="Path to CSV files produced by grid_search_parallel.py")
    parser.add_argument("-k", "--kind", choices=["cindex", "rmse"], default="cindex",
                        help="Which performance measure to plot")

    args = parser.parse_args()

    dists = []
    names = []
    datasets = []
    for filename in args.files:
        _df = pandas.read_csv(filename, index_col=0, comment="#")
        best_perf, c_dist, rmse_dist = format_data(_df)
        m = method_pat.search(basename(filename))
        if m is None:
            raise ValueError('Could not find method in filename %s' % basename(filename))

        datasets.append(m.group(1))
        names.append(m.group(2))

        if args.kind == "cindex":
            dist = c_dist
        else:
            dist = rmse_dist

        if pandas.isnull(dist).any():
            print("!!! %s has %d missing values" % (filename, pandas.isnull(dist).sum()))
            dist[pandas.isnull(dist)] = dist.mean(skipna=True)

        dists.append(dist.values)
        print("Best performance for {}:".format(filename))
        print(best_perf)
        print()

    dfa = pandas.DataFrame(dists)
    data_cols = dfa.columns.tolist()
    dfa['Method'] = [METHODS[v] for v in names]
    dfa['Data'] = [DATASETS[v] for v in datasets]

    if args.kind == "cindex":
        _g = plot_cindex()
    else:
        _g = plot_rmse()

    if isinstance(_g, seaborn.FacetGrid):
        _g.fig.tight_layout()

    if args.output is None:
        plt.show()
    else:
        plt.savefig(args.output)
