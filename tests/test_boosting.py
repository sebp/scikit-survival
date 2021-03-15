import numpy
from numpy.testing import assert_array_equal, assert_array_almost_equal
import pandas
import pytest
from sklearn.metrics import mean_squared_error

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.testing import assert_cindex_almost_equal
from sksurv.util import Surv
from sksurv.linear_model.coxph import BreslowEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sksurv


def early_stopping_monitor(i, est, locals_):
    """Returns True on the 10th iteration. """
    return i == 9


class TestGradientBoosting(object):

    @staticmethod
    def test_fit(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_depth=3, min_samples_split=10,
                                                 random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.max_features_ == 14
        assert not hasattr(model, "oob_improvement_")

        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.86272605091218779, 64826, 10309, 14, 14))

        assert (100,) == model.train_score_.shape

        with pytest.raises(ValueError, match="Number of features of the model must match the input. "
                                             "Model n_features is 14 and input n_features is 2 "):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    def test_fit_subsample(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=50, max_features=8, subsample=0.6,
                                                 random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.max_features_ == 8
        assert hasattr(model, "oob_improvement_")

        incl_mask = numpy.ones(whas500_data.x.shape[0], dtype=bool)
        incl_mask[[35, 111, 174, 206, 236, 268, 497]] = False
        x_test = whas500_data.x[incl_mask]
        y_test = whas500_data.y[incl_mask]

        p = model.predict(x_test)

        assert_cindex_almost_equal(y_test['fstat'], y_test['lenfol'], p,
                                   (0.8330510326740247, 60985, 12221, 2, 14))

        assert (50,) == model.train_score_.shape
        assert (50,) == model.oob_improvement_.shape

        with pytest.raises(ValueError, match="Number of features of the model must match the input. "
                                             "Model n_features is 14 and input n_features is 2 "):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    @pytest.mark.slow
    def test_fit_dropout(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=100, max_features=8,
                                                 learning_rate=1.0, dropout_rate=0.03,
                                                 random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert not hasattr(model, "oob_improvement_")
        assert model.max_features_ == 8

        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.9094333, 68343, 6806, 0, 14))

    @staticmethod
    def test_fit_int_param_as_float(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        # Account for https://github.com/scikit-learn/scikit-learn/pull/12344
        max_depth = 4

        model = GradientBoostingSurvivalAnalysis(
            n_estimators=100.0,
            max_depth=float(max_depth),
            min_samples_split=10.0,
            random_state=0)
        params = model.get_params()
        assert 100 == params["n_estimators"]
        assert max_depth == params["max_depth"]
        assert 10 == params["min_samples_split"]

        model.set_params(max_leaf_nodes=15.0)
        assert 15 == model.get_params()["max_leaf_nodes"]

        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.90256690042449006, 67826, 7321, 2, 14))

    @staticmethod
    def test_max_features(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=10, max_features="auto", max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert model.max_features_ == whas500_data.x.shape[1]

        model.set_params(max_features="sqrt")
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - int(numpy.sqrt(whas500_data.x.shape[1]))), 7) == 0

        model.set_params(max_features="log2")
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - int(numpy.log2(whas500_data.x.shape[1]))), 7) == 0

        model.set_params(max_features=0.25)
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - int(0.25 * whas500_data.x.shape[1])), 7) == 0

        model.set_params(max_features=5)
        model.fit(whas500_data.x, whas500_data.y)
        assert round(abs(model.max_features_ - 5), 7) == 0

        model.set_params(max_features=-1)
        with pytest.raises(ValueError,
                           match=r"max_features must be in \(0, n_features\]"):
            model.fit(whas500_data.x, whas500_data.y)

        model.set_params(max_features=-1.125)
        with pytest.raises(ValueError,
                           match=r"max_features must be in \(0, 1.0\]"):
            model.fit(whas500_data.x, whas500_data.y)

        model.set_params(max_features="fail_me")
        with pytest.raises(ValueError,
                           match="Invalid value for max_features: 'fail_me'. "
                                 "Allowed string values are 'auto', 'sqrt' "
                                 "or 'log2'"):
            model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    @pytest.mark.parametrize('presort', ['auto', True, False, None])
    def test_presort(make_whas500, presort):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=10, presort=presort, random_state=0)

        with pytest.deprecated_call(match="The parameter 'presort' is deprecated "):
            model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_ccp_alpha(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        est_full = GradientBoostingSurvivalAnalysis(
            n_estimators=10,
            max_leaf_nodes=20,
            random_state=1)
        est_full.fit(whas500_data.x, whas500_data.y)

        est_pruned = GradientBoostingSurvivalAnalysis(
            n_estimators=10,
            max_leaf_nodes=20,
            ccp_alpha=10.0,
            random_state=1)
        est_pruned.fit(whas500_data.x, whas500_data.y)

        tree = est_full.estimators_[0, 0].tree_
        subtree = est_pruned.estimators_[0, 0].tree_
        assert tree.node_count > subtree.node_count
        assert tree.max_depth > subtree.max_depth

    @staticmethod
    def test_negative_ccp_alpha(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        clf = GradientBoostingSurvivalAnalysis()
        msg = "ccp_alpha must be greater than or equal to 0"

        with pytest.raises(ValueError, match=msg):
            clf.set_params(ccp_alpha=-1.0)
            clf.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_fit_verbose(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=10, verbose=1, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_ipcwls_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 590.5441693629117), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 392.97741487479743), 7) == 0

        cindex = model.score(whas500_data.x, whas500_data.y)
        assert round(abs(cindex - 0.8979161399), 7) == 0

    @staticmethod
    def test_squared_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 580.23345259002951), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 383.10639243317951), 7) == 0

        cindex = model.score(whas500_data.x, whas500_data.y)
        assert round(abs(cindex - 0.9021810004), 7) == 0

    @staticmethod
    def test_ipcw_loss_staged_predict(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        # Test whether staged decision function eventually gives
        # the same prediction.
        model = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

        model.set_params(dropout_rate=0.03)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

    @staticmethod
    def test_squared_loss_staged_predict(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        # Test whether staged decision function eventually gives
        # the same prediction.
        model = GradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, max_depth=3, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

        model.set_params(dropout_rate=0.03)
        model.fit(whas500_data.x, whas500_data.y)

        y_pred = model.predict(whas500_data.x)

        # test if prediction for last stage equals ``predict``
        for y in model.staged_predict(whas500_data.x):
            assert y.shape == y_pred.shape

        assert_array_equal(y_pred, y)

    @staticmethod
    def test_monitor_early_stopping(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        est = GradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=50, max_depth=1,
                                               subsample=0.5,
                                               random_state=0)
        est.fit(whas500_data.x, whas500_data.y, monitor=early_stopping_monitor)

        assert est.n_estimators == 50  # this is not altered
        assert est.estimators_.shape[0] == 10
        assert est.train_score_.shape[0] == 10
        assert est.oob_improvement_.shape[0] == 10

    @staticmethod
    def test_predict_cumulative_hazard_function(rossi):

        model = GradientBoostingSurvivalAnalysis(n_estimators=50,
                                                 max_features=4,
                                                 subsample=0.6,
                                                 random_state=0)
        model.fit(rossi.x.values, rossi.y)

        test_idx = [9, 3, 313, 122, 431]
        f = model.predict_cumulative_hazard_function(rossi.x.values[test_idx, :])
        assert len(f) == len(test_idx)

        expected_x = numpy.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52])
        assert_array_almost_equal(f[0].x, expected_x)

        expected_y = numpy.array([
            [0.002349782713082724, 0.004705133887878451, 0.0070720578141535255, 0.009444605656360844,
             0.01182840533804299, 0.01422337435600477, 0.016635890115468087, 0.02872354585363421,
             0.03363612246511115, 0.0361028220843282, 0.0410712648694533, 0.046172518547257575,
             0.0487324669519664, 0.05643986464269394, 0.06171429059123936, 0.0670273634607806,
             0.07504016194927861, 0.08315724133875178, 0.0886244527860999, 0.10240220852560565,
             0.1079736668069592, 0.11077417976548282, 0.11359662801563679, 0.12490955905147934,
             0.13346445064079668, 0.1422839861987567, 0.14820224875222532, 0.15416022967387014,
             0.16018863488244675, 0.16323165651225088, 0.1693649797807942, 0.1755490636171977,
             0.18183591923735268, 0.1945041128142993, 0.2041530065817895, 0.2171718704163144,
             0.22047757699160772, 0.2271057475484405, 0.2404374670312662, 0.24721195504625837,
             0.26082719431223406, 0.2677632608127352, 0.27474016958023556, 0.2889098516370246,
             0.2924921647175684, 0.2996860220322564, 0.31791175071788946, 0.32904720847649926,
             0.34420765578328727],
            [0.0013211373694091235, 0.0026454055401549884, 0.003976180352711234, 0.005310117145079116,
             0.0066503801503014996, 0.007996922981938624, 0.009353331260189165, 0.016149471863033785,
             0.018911509606086556, 0.020298382114727814, 0.02309182994913991, 0.025959949127589966,
             0.02739924965626869, 0.03173264229441568, 0.03469812552144256, 0.037685337519923874,
             0.04219043812252882, 0.04675416942082613, 0.04982804400052165, 0.057574423217931124,
             0.06070690933957066, 0.06228146442605613, 0.0638683523692438, 0.0702289132269718,
             0.0750387992249258, 0.07999747814514277, 0.08332495084201368, 0.08667475471876664,
             0.09006415381283972, 0.0917750565140432, 0.09522344453886983, 0.09870037208895517,
             0.10223508185153146, 0.10935762294617421, 0.11478259865082836, 0.12210229992502625,
             0.12396089411103588, 0.1276875041353138, 0.13518310477497203, 0.13899198004900903,
             0.14664698631302595, 0.15054670716785215, 0.1544693911012999, 0.16243612622691106,
             0.16445023912904425, 0.16849489980158913, 0.17874209887971682, 0.1850028774991475,
             0.19352665859705126],
            [0.0023641747237483418, 0.004733952015069738, 0.007115372938961655, 0.009502452224293322,
             0.011900852264660741, 0.014310490051550993, 0.01673778205068441, 0.02889947257910473,
             0.03384213786839617, 0.03632394559400578, 0.04132281922754801, 0.04645531719740118,
             0.049030944841104686, 0.056785549002946134, 0.062092279893596995, 0.06743789441091694,
             0.07549976989732878, 0.08366656498710758, 0.08916726215423028, 0.10302940425271603,
             0.10863498674758171, 0.11145265236117587, 0.11429238761624871, 0.1256746083882106,
             0.13428189720145473, 0.1431554508816476, 0.1491099617645116, 0.15510443428365286,
             0.1611697624262418, 0.16423142203457392, 0.17040231084200944, 0.17662427111695983,
             0.18294963263497283, 0.1956954166955239, 0.2054034082601168, 0.2185020103726184,
             0.2218279638260695, 0.2284967307754182, 0.24191010472268423, 0.24872608529918988,
             0.2624247155389947, 0.2694032642411466, 0.27642290536208364, 0.29067937425841933,
             0.2942836283838843, 0.30152154682414933, 0.3198589049298866, 0.33106256543164203,
             0.3463158678429127],
            [0.0010452140870053051, 0.0020929050986185646, 0.003145743821466689, 0.004201084135684852,
             0.0052614294152805935, 0.0063267429617535125, 0.00739986152836567, 0.012776608912734126,
             0.014961787248254426, 0.016059007504434564, 0.018269035844748514, 0.02053814020735302,
             0.021676838743056398, 0.025105190052157625, 0.027451323705959815, 0.029814648015740238,
             0.033378845594471046, 0.036989428681997186, 0.03942131584738878, 0.04554984181962945,
             0.04802809934038264, 0.04927380413631657, 0.05052926603689126, 0.055561405739914894,
             0.05936673342071433, 0.06328977820042946, 0.06592229879777696, 0.06857248664482356,
             0.07125400013587911, 0.07260757596092227, 0.07533575837742693, 0.0780865197584871,
             0.08088299537326159, 0.08651797358201845, 0.09080992774171501, 0.09660088866796906,
             0.09807131019280434, 0.10101960715596764, 0.10694973036688, 0.10996311124933147,
             0.11601934776835661, 0.11910460087468251, 0.12220802115560508, 0.12851088108034778,
             0.13010434079685726, 0.13330426262935455, 0.14141130514954084, 0.1463645024174202,
             0.15310806768502228],
            [0.0013086996562767508, 0.0026205006392801063, 0.003938747007977584, 0.005260125588349988,
             0.006587770824090652, 0.007921636765460231, 0.00926527527620143, 0.015997434305908616,
             0.018733469126097854, 0.020107285064837246, 0.022874434269281586, 0.025715551832004906,
             0.027141302212529854, 0.03143389856728406, 0.03437146355467086, 0.03733055275020754,
             0.041793240542286406, 0.04631400705734785, 0.049358942958059336, 0.05703239467772964,
             0.060135390327996384, 0.061695121926076874, 0.06326707027443694, 0.06956775028014968,
             0.07433235409653255, 0.07924435003937468, 0.0825404965836306, 0.08585876407315124,
             0.08921625401485354, 0.09091104959692793, 0.09432697312410653, 0.09777116749409165,
             0.10127260009181659, 0.10832808674915403, 0.11370198957287474, 0.12095278026534874,
             0.12279387690579623, 0.1264854031397648, 0.13391043721104823, 0.13768345421696537,
             0.14526639320460139, 0.14912940053484508, 0.15301515476014674, 0.16090688786977655,
             0.16290203911124296, 0.1669086217380617, 0.17705934960472022, 0.1832611867618497,
             0.19170472159123234]
        ])

        for i, ff in enumerate(f):
            actual_y = ff(expected_x)
            # check that values increase
            assert (numpy.diff(actual_y) > 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])

        pipe = make_pipeline(StandardScaler(with_mean=False, with_std=False),
                             GradientBoostingSurvivalAnalysis(n_estimators=50, max_features=4,
                             subsample=0.6, random_state=0))
        pipe.fit(rossi.x, rossi.y)
        f = pipe.predict_cumulative_hazard_function(rossi.x.values[test_idx, :])
        assert len(f) == len(test_idx)

        for i, ff in enumerate(f):
            actual_y = [ff(v) for v in expected_x]
            # check that values increase
            assert (numpy.diff(actual_y) > 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])

    @staticmethod
    def test_predict_survival_function(rossi):
        # whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = GradientBoostingSurvivalAnalysis(n_estimators=50, max_features=4, subsample=0.6,
                                                 random_state=0)
        # model.fit(whas500_data.x, whas500_data.y)
        model.fit(rossi.x.values, rossi.y)

        test_idx = [9, 3, 313, 122, 431]

        f = model.predict_survival_function(rossi.x.values[test_idx, :])
        assert len(f) == len(test_idx)

        expected_x = numpy.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52])

        assert_array_almost_equal(f[0].x, expected_x)

        expected_y = numpy.array([
            [0.9976529758652071, 0.9953059179143756, 0.992952890340469, 0.9905998545518362, 0.9882412752410719,
             0.9858772999583348, 0.9835017221469433, 0.9716850537006477, 0.9669232822920193, 0.9645411122566178,
             0.9597607302874311, 0.9548772139502996, 0.9524359036971258, 0.9451233181147078, 0.9401514585398891,
             0.9351696115096774, 0.9277062270899182, 0.9202064413507846, 0.9151892058017808, 0.9026664185168419,
             0.8976512341424373, 0.895140867027512, 0.8926179403435786, 0.882576720050861, 0.8750585761963418,
             0.8673748990060419, 0.8622567069626608, 0.8571346715991324, 0.8519830600827656, 0.8493943978778279,
             0.8442007309533722, 0.8389962319736594, 0.833738129581343, 0.823242792404832, 0.815337609622213,
             0.8047916375439823, 0.8021356249538282, 0.796836514326755, 0.786283812571885, 0.7809751443646552,
             0.7704140400408184, 0.7650888861857841, 0.7597695088108667, 0.7490797306704354, 0.7464010932763434,
             0.7410508578006687, 0.7276670016716799, 0.7196090442985316, 0.7087817301414391],
            [0.9986797349483725, 0.9973580904616206, 0.996031714185552, 0.9947039566048617, 0.9933716846874883,
             0.9920349673418764, 0.990690275081885, 0.9839802317043528, 0.9812661910327708, 0.9799062431838895,
             0.9771727459294697, 0.9743741133605702, 0.9729727049499409, 0.968765554394405, 0.9658969518904611,
             0.9630159181980773, 0.9586871925981301, 0.9543219702792919, 0.951393008171331, 0.9440516284118139,
             0.9410990266940137, 0.9396183804201115, 0.9381284937952404, 0.9321804070552691, 0.9277074912986475,
             0.9231186743550132, 0.9200521269259667, 0.9169752890232895, 0.9138725549817183, 0.9123103447347837,
             0.909169762751879, 0.9060141344805381, 0.9028172907602041, 0.8964097834608675, 0.8915599491751227,
             0.8850578225223946, 0.8834143869121938, 0.8801283726520858, 0.8735559447784347, 0.8702350076723773,
             0.8635987858038306, 0.8602375498094372, 0.8568697195791439, 0.850070385694277, 0.8483599710250682,
             0.8449355727470126, 0.8363215598535066, 0.8311018923536981, 0.8240478679422283],
            [0.9976386177362593, 0.995277235475128, 0.992909881393571, 0.9905425534074092, 0.9881696827912816,
             0.9857914183124997, 0.9834015163583528, 0.9715141233670094, 0.9667241017199096, 0.9643278531198206,
             0.9595193286570882, 0.9546072141427315, 0.9521516650606351, 0.94479666022864, 0.9397961585000025,
             0.9347857742344572, 0.9272799439035663, 0.919737877784217, 0.9146925673294617, 0.9021004475019965,
             0.8970577957295306, 0.8945337444612501, 0.8919971088416895, 0.8819017635370093, 0.8743435548585383,
             0.8666193416833108, 0.8614743804475145, 0.8563257430481355, 0.8511475659658383, 0.8485456270006028,
             0.8433254693597584, 0.8380946217286782, 0.8328101011311047, 0.8222626440142602, 0.8143187472331084,
             0.8037218636626045, 0.801053162602385, 0.7957288986167081, 0.7851267535667966, 0.7797935410481937,
             0.7691842698120197, 0.7638351661230022, 0.7584920925508857, 0.7477553892134351, 0.7450651398505458,
             0.7396918881718925, 0.7262515003513512, 0.7181602356298156, 0.7072890419547924],
            [0.998955331958977, 0.9979092835001478, 0.996859198846481, 0.995807728073676, 0.9947523876613199,
             0.993693228735703, 0.9926274500381619, 0.9873046654497818, 0.9851495841602987, 0.9840692508713584,
             0.9818968313766419, 0.9796713308942587, 0.9785564154810429, 0.9752073245235037, 0.9729220396401406,
             0.9706254242247994, 0.9671720812919782, 0.9636863227297788, 0.9613455936781419, 0.9554719789231052,
             0.9531070050230722, 0.9519204542553067, 0.9507261042799326, 0.9459539348856635, 0.9423611104650875,
             0.9386714277574166, 0.9362036056112198, 0.9337257750018043, 0.9312253307250115, 0.9299656993253012,
             0.927432040969035, 0.9248844023066343, 0.9223015986859562, 0.9171190647218368, 0.9131912667309825,
             0.9079182943116229, 0.9065842527509536, 0.9039153095036572, 0.898570842681558, 0.8958671821081097,
             0.8904579947557925, 0.8877149401461759, 0.884964258066226, 0.8794039935008445, 0.8780038145233294,
             0.8751987613172462, 0.8681321710233795, 0.8638427730126911, 0.8580369907296157],
            [0.9986921563176735, 0.9973829298753106, 0.9960689996819307, 0.9947536846471178, 0.9934338809663568,
             0.9921096567126866, 0.9907775151298321, 0.9841298450281745, 0.9814409116896323, 0.9800935182694196,
             0.9773852021556062, 0.974612276860836, 0.9732237131440796, 0.969055010261251, 0.9662125252285619,
             0.9633576421944986, 0.9590680564653353, 0.9547421193533706, 0.951839412319395, 0.9445634700416912,
             0.9416370364062885, 0.9401694801611246, 0.9386927432807441, 0.9327969339900912, 0.9283630972840108,
             0.9238141628367171, 0.9207741488944463, 0.9177238376441947, 0.9146477559363957, 0.913098927804747,
             0.9099851728966868, 0.9068563982343815, 0.9036866542410497, 0.897333145015816, 0.8925238976816932,
             0.8860757988691516, 0.8844459485090663, 0.8811870120132949, 0.8746683988365732, 0.8713744780104642,
             0.8647918877406029, 0.8614576346076446, 0.8581167171598264, 0.851371340415744, 0.8496744191991017,
             0.8462769391199105, 0.8377300640993981, 0.832550676121757, 0.8255505997730772]
        ])

        for i, ff in enumerate(f):
            actual_y = [ff(v) for v in expected_x]
            # check that values decrease
            assert (numpy.diff(actual_y) < 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])

        pipe = make_pipeline(StandardScaler(with_mean=False, with_std=False),
                             GradientBoostingSurvivalAnalysis(n_estimators=50, max_features=4,
                                                              subsample=0.6, random_state=0))
        pipe.fit(rossi.x.values, rossi.y)
        f = pipe.predict_survival_function(rossi.x.values[test_idx, :])
        assert len(f) == len(test_idx)

        for i, ff in enumerate(f):

            actual_y = ff(expected_x)
            # check that values decrease
            assert (numpy.diff(actual_y) < 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])

    @staticmethod
    def test_predict_survival_function_with_breslow(rossi):

        baseline_estimator = BreslowEstimator()
        model = GradientBoostingSurvivalAnalysis(n_estimators=50,
                                                 max_features=4,
                                                 subsample=0.6,
                                                 random_state=0)
        rossi_train_x = rossi.x.values[:300]
        rossi_train_y = rossi.y[:300]
        rossi_test_x = rossi.x.values[300:]
        model.fit(rossi_train_x, rossi_train_y)

        f_implementation = model.predict_survival_function(rossi_test_x)
        assert len(f_implementation) == len(rossi_test_x)

        train_risks = model.predict(rossi_train_x)
        _, event, time = sksurv.util.check_arrays_survival(rossi_train_x, rossi_train_y)
        f = baseline_estimator.fit(train_risks, event, time).get_survival_function(model.predict(rossi_test_x))

        assert len(f) == len(rossi_test_x)

        for true, implementation in zip(f, f_implementation):
            y_true = true.y
            y_implement = implementation.y
            assert (numpy.allclose(y_true, y_implement))

        pipe = make_pipeline(StandardScaler(with_mean=False, with_std=False),
                             GradientBoostingSurvivalAnalysis(n_estimators=50,
                                                              max_features=4,
                                                              subsample=0.6,
                                                              random_state=0))
        pipe.fit(rossi_train_x, rossi_train_y)
        f_pipe = pipe.predict_survival_function(rossi_test_x)
        assert len(f_pipe) == len(rossi_test_x)

        for true, pipe_ in zip(f, f_pipe):
            y_true = true.y
            y_pipe = pipe_.y
            assert (numpy.allclose(y_true, y_pipe))


class TestSparseGradientBoosting(object):

    @staticmethod
    @pytest.mark.parametrize('loss', ['coxph', 'squared', 'ipcwls'])
    def test_fit(whas500_sparse_data, loss):
        model = GradientBoostingSurvivalAnalysis(loss=loss, n_estimators=100, max_depth=1, min_samples_split=10,
                                                 subsample=0.5, random_state=0)
        model.fit(whas500_sparse_data.x_sparse, whas500_sparse_data.y)

        assert model.estimators_.shape[0] == 100
        assert model.train_score_.shape == (100,)
        assert model.oob_improvement_.shape == (100,)

        sparse_predict = model.predict(whas500_sparse_data.x_dense)

        model.fit(whas500_sparse_data.x_sparse, whas500_sparse_data.y)
        dense_predict = model.predict(whas500_sparse_data.x_dense)

        assert_array_almost_equal(sparse_predict, dense_predict)

    @staticmethod
    @pytest.mark.parametrize('loss', ['coxph', 'squared', 'ipcwls'])
    @pytest.mark.slow
    def test_dropout(whas500_sparse_data, loss):
        model = GradientBoostingSurvivalAnalysis(loss=loss, n_estimators=100, max_depth=1, min_samples_split=10,
                                                 dropout_rate=0.03, random_state=0)
        model.fit(whas500_sparse_data.x_sparse, whas500_sparse_data.y)

        assert model.estimators_.shape[0] == 100
        assert model.train_score_.shape == (100,)

        sparse_predict = model.predict(whas500_sparse_data.x_dense)

        model.fit(whas500_sparse_data.x_dense, whas500_sparse_data.y)
        dense_predict = model.predict(whas500_sparse_data.x_dense)

        assert_array_almost_equal(sparse_predict, dense_predict)


class TestComponentwiseGradientBoosting(object):

    @staticmethod
    def test_fit(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100)
        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7755659, 58283, 16866, 0, 14))

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.040919
        expected_coef['hr'] = 0.004977
        expected_coef['diasbp'] = -0.003407
        expected_coef['bmi'] = -0.017938
        expected_coef['sho'] = 0.429904
        expected_coef['chf'] = 0.508211

        assert_array_almost_equal(expected_coef.values, model.coef_)

        assert (100,) == model.train_score_.shape

        with pytest.raises(ValueError, match='Dimensions of X are inconsistent with training data: '
                                             'expected 14 features, but got 2'):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    def test_fit_subsample(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, subsample=0.6, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7750602, 58245, 16904, 0, 14))

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.041299
        expected_coef['hr'] = 0.00487
        expected_coef['diasbp'] = -0.003381
        expected_coef['bmi'] = -0.017018
        expected_coef['sho'] = 0.433685
        expected_coef['chf'] = 0.510277

        assert_array_almost_equal(expected_coef.values, model.coef_)

        assert (100,) == model.train_score_.shape
        assert (100,) == model.oob_improvement_.shape

        with pytest.raises(ValueError, match='Dimensions of X are inconsistent with training data: '
                                             'expected 14 features, but got 2'):
            model.predict(whas500_data.x[:, :2])

    @staticmethod
    def test_fit_dropout(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0,
                                                              dropout_rate=0.03, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)
        p = model.predict(whas500_data.x)

        assert_cindex_almost_equal(whas500_data.y['fstat'], whas500_data.y['lenfol'], p,
                                   (0.7772425, 58409, 16740, 0, 14))

        expected_coef = pandas.Series(numpy.zeros(15, dtype=float), index=whas500_data.names)
        expected_coef['age'] = 0.275537
        expected_coef['hr'] = 0.040048
        expected_coef['diasbp'] = -0.029998
        expected_coef['bmi'] = -0.138909
        expected_coef['sho'] = 3.318941
        expected_coef['chf'] = 2.851386
        expected_coef['mitype'] = -0.075817

        assert_array_almost_equal(expected_coef.values, model.coef_)

    @staticmethod
    def test_feature_importances(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=100, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        assert whas500_data.x.shape[1] + 1 == len(model.feature_importances_)

    @staticmethod
    def test_fit_verbose(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(n_estimators=10, verbose=1, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

    @staticmethod
    def test_ipcwls_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss="ipcwls", n_estimators=100, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 806.283308322), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 542.884585289), 7) == 0

        cindex = model.score(whas500_data.x, whas500_data.y)
        assert round(abs(cindex - 0.7773356931), 7) == 0

    @staticmethod
    def test_squared_loss(make_whas500):
        whas500_data = make_whas500(with_std=False, to_numeric=True)

        model = ComponentwiseGradientBoostingSurvivalAnalysis(loss="squared", n_estimators=100, random_state=0)
        model.fit(whas500_data.x, whas500_data.y)

        time_predicted = model.predict(whas500_data.x)
        time_true = whas500_data.y["lenfol"]
        event_true = whas500_data.y["fstat"]

        rmse_all = numpy.sqrt(mean_squared_error(time_true, time_predicted))
        assert round(abs(rmse_all - 793.6256945839657), 7) == 0

        rmse_uncensored = numpy.sqrt(mean_squared_error(time_true[event_true], time_predicted[event_true]))
        assert round(abs(rmse_uncensored - 542.83358120153525), 7) == 0

        cindex = model.score(whas500_data.x, whas500_data.y)
        assert round(abs(cindex - 0.7777082862), 7) == 0


@pytest.fixture(params=[GradientBoostingSurvivalAnalysis, ComponentwiseGradientBoostingSurvivalAnalysis])
def sample_gb_class(request):
    x = numpy.arange(100).reshape(5, 20)
    y = Surv.from_arrays([False, False, True, True, False], [12, 14, 6, 9, 1])

    return request.param, x, y


def test_param_n_estimators(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(n_estimators=0)

    with pytest.raises(ValueError, match="n_estimators must be greater than 0 but was 0"):
        model.fit(x, y)

    model.set_params(n_estimators=-1)
    with pytest.raises(ValueError, match="n_estimators must be greater than 0 but was -1"):
        model.fit(x, y)


def test_param_learning_rate(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(learning_rate=0)

    with pytest.raises(ValueError, match="learning_rate must be within ]0; 1] but was 0"):
        model.fit(x, y)

    model.set_params(learning_rate=1.2)
    with pytest.raises(ValueError, match="learning_rate must be within ]0; 1] but was 1.2"):
        model.fit(x, y)


def test_param_subsample(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(subsample=0)

    with pytest.raises(ValueError, match="subsample must be in ]0; 1] but was 0"):
        model.fit(x, y)

    model.set_params(subsample=1.2)
    with pytest.raises(ValueError, match="subsample must be in ]0; 1] but was 1.2"):
        model.fit(x, y)


def test_param_dropout_rate(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(dropout_rate=-0.1)

    with pytest.raises(ValueError, match=r"dropout_rate must be within \[0; 1\[, but was -0.1"):
        model.fit(x, y)

    model.set_params(dropout_rate=1.2)
    with pytest.raises(ValueError, match=r"dropout_rate must be within \[0; 1\[, but was 1.2"):
        model.fit(x, y)


def test_param_sample_weight(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls()

    with pytest.raises(ValueError, match=r"Found input variables with inconsistent numbers of samples: \[5, 3\]"):
        model.fit(x, y, [2, 3, 4])

    model.set_params(dropout_rate=1.2)
    with pytest.raises(ValueError, match=r"Found input variables with inconsistent numbers of samples: \[5, 8\]"):
        model.fit(x, y, [2, 4, 5, 6, 7, 1, 2, 7])


def test_param_loss(sample_gb_class):
    est_cls, x, y = sample_gb_class
    model = est_cls(loss="")

    with pytest.raises(ValueError, match="Loss '' not supported"):
        model.fit(x, y)

    model.set_params(loss="unknown")
    with pytest.raises(ValueError, match="Loss 'unknown' not supported"):
        model.fit(x, y)

    model.set_params(loss=None)
    with pytest.raises(ValueError, match="Loss None not supported"):
        model.fit(x, y)
