import warnings

import numpy
from numpy.testing import assert_array_almost_equal
import pandas
import pytest
from scipy.optimize import check_grad
from sksurv.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sksurv.column import standardize
from sksurv.linear_model.coxph import CoxPHSurvivalAnalysis, CoxPHOptimizer
from sksurv.preprocessing import OneHotEncoder


@pytest.fixture
def coef_rossi_coxph_breslow():
    return pandas.Series({"fin": -0.37902189,
                          "age": -0.05724593,
                          "race": 0.31412977,
                          "wexp": -0.15111460,
                          "mar": -0.43278257,
                          "paro": -0.08498284,
                          "prio": 0.09111154})


@pytest.fixture
def coef_rossi_coxph_efron():
    return pandas.Series({"fin": -0.379422166485887,
                          "age": -0.0574377426840626,
                          "race": 0.313899787842507,
                          "wexp": -0.149795697666534,
                          "mar": -0.433703877937307,
                          "paro": -0.0848710825003959,
                          "prio": 0.0914970809852961})


def assert_gradient_correctness(cph):
    def grad(x):
        cph.update(x)
        return cph.gradient

    rnd = numpy.random.RandomState(9)
    coef = rnd.randn(cph.x.shape[1])

    err = check_grad(cph.nlog_likelihood,
                     grad,
                     coef)

    assert round(err, 4) == 0


class TestCoxPH(object):

    @staticmethod
    def test_likelihood_breslow(rossi, coef_rossi_coxph_breslow):
        cph = CoxPHOptimizer(rossi.x.values, rossi.y['arrest'], rossi.y['week'],
                             alpha=numpy.zeros(rossi.x.shape[1]),
                             ties="breslow")

        w = coef_rossi_coxph_breslow.loc[rossi.x.columns].values

        actual_loss = cph.nlog_likelihood(w)

        assert round(abs(659.1206 - rossi.x.shape[0] * actual_loss), 4) == 0

    @staticmethod
    def test_gradient_breslow(rossi):
        cph = CoxPHOptimizer(rossi.x.values, rossi.y['arrest'], rossi.y['week'],
                             alpha=numpy.zeros(rossi.x.shape[1]),
                             ties="breslow")

        assert_gradient_correctness(cph)

    @staticmethod
    def test_fit_breslow(rossi, coef_rossi_coxph_breslow):
        cph = CoxPHSurvivalAnalysis()
        cph.fit(rossi.x.values, rossi.y)

        actual = pandas.Series(cph.coef_, index=rossi.x.columns)
        assert_array_almost_equal(coef_rossi_coxph_breslow.values,
                                  actual.loc[coef_rossi_coxph_breslow.index].values)

    @staticmethod
    def test_likelihood_efron(rossi, coef_rossi_coxph_efron):
        cph = CoxPHOptimizer(rossi.x.values, rossi.y['arrest'], rossi.y['week'],
                             alpha=numpy.zeros(rossi.x.shape[1]),
                             ties="efron")

        w = coef_rossi_coxph_efron.loc[rossi.x.columns].values

        actual_loss = cph.nlog_likelihood(w)

        assert round(abs(658.7477 - rossi.x.shape[0] * actual_loss), 4) == 0

    @staticmethod
    def test_gradient_efron(rossi):
        cph = CoxPHOptimizer(rossi.x.values.astype(float), rossi.y['arrest'], rossi.y['week'],
                             alpha=numpy.zeros(rossi.x.shape[1]), ties="efron")

        assert_gradient_correctness(cph)

    @staticmethod
    def test_fit_efron(rossi, coef_rossi_coxph_efron):
        cph = CoxPHSurvivalAnalysis(ties="efron")
        cph.fit(rossi.x.values, rossi.y)

        actual = pandas.Series(cph.coef_, index=rossi.x.columns)
        assert_array_almost_equal(coef_rossi_coxph_efron.values,
                                  actual.loc[coef_rossi_coxph_efron.index].values)

    @staticmethod
    def test_predict(rossi):
        cph = CoxPHSurvivalAnalysis()
        xc = standardize(rossi.x, with_std=False)
        cph.fit(xc.values, rossi.y)

        expected = numpy.array([-0.136002823953217, -1.13104636905577, 0.741965816026403, -0.98072115186145,
                                -0.600098931134794, -0.997407014712788, -0.0993800739865776, -0.266761246895696,
                                -0.665145743277517, -0.418747210463951, -0.0770761787926419, 0.411385264707043,
                                -0.0770761787926419, 0.563114305747799, -1.07096133044073])

        idx = numpy.array([15, 77, 79, 90, 113, 122, 134, 172, 213, 219, 257, 313, 364, 395, 409])

        pred = cph.predict(xc.iloc[idx, :].values)

        assert_array_almost_equal(expected, pred)

    @staticmethod
    def test_fit_ridge_1(rossi):
        # coxph(Surv(week, arrest) ~ ridge(fin, age, race, wexp, mar, paro, prio,
        #     theta=1, scale=FALSE), data=rossi, ties="breslow")
        cph = CoxPHSurvivalAnalysis(alpha=1.0)
        cph.fit(rossi.x.values, rossi.y)

        expected = pandas.Series({'fin': -0.36366779384675196,
                                  'age': -0.057788417088377418,
                                  'race': 0.28960521422300672,
                                  'wexp': -0.15082851149160476,
                                  'mar': -0.3829568076550468,
                                  'paro': -0.08230383874483703,
                                  'prio': 0.090951189830228568})

        actual = pandas.Series(cph.coef_, index=rossi.x.columns)
        assert_array_almost_equal(expected.values,
                                  actual.loc[expected.index].values)

    @staticmethod
    def test_fit_ridge_2(rossi):
        # coxph(Surv(week, arrest) ~ ridge(fin, age, race, wexp, mar, paro, prio,
        #     theta=19.67, scale=FALSE), data=rossi, ties="breslow")
        cph = CoxPHSurvivalAnalysis(alpha=19.67)
        cph.fit(rossi.x.values, rossi.y)

        expected = pandas.Series({'fin': -0.21145000,
                                  'age': -0.06223214,
                                  'race': 0.11957591,
                                  'wexp': -0.10694088,
                                  'mar': -0.13696844,
                                  'paro': -0.04929119,
                                  'prio': 0.09029133})

        actual = pandas.Series(cph.coef_, index=rossi.x.columns)
        assert_array_almost_equal(expected.values,
                                  actual.loc[expected.index].values)

    @staticmethod
    def test_fit_unpenalized():
        X, y = load_breast_cancer()
        included = X["grade"] != "unkown"
        X = X.loc[included, :]
        y = y[included.values]

        X["grade"] = pandas.Series(pandas.Categorical(
            X["grade"].astype(object),
            categories=["intermediate", "poorly differentiated",
                        "well differentiated"]),
            index=X.index, name="grade")

        enc = OneHotEncoder()
        X = enc.fit_transform(X)

        cols_unpen = ['age', 'size', 'grade=poorly differentiated',
                      'grade=well differentiated', 'er=positive']
        X = pandas.concat((
            X.loc[:, cols_unpen],
            X.drop(cols_unpen, axis=1)),
            axis=1)

        alphas = numpy.ones(X.shape[1])
        alphas[:len(cols_unpen)] = 0.0

        cph = CoxPHSurvivalAnalysis(alpha=alphas)
        cph.fit(X, y)

        coef = numpy.array([
            -0.0228825990482334, 0.635554486750423, -0.242079636336473,
            -1.30197563647684, -2.27790151300312,
            0.291950212930807, 0.210861165049552, -0.612456645638769, -0.453414844486013, -0.1239424190253,
            0.196855946938761, 1.08724198521351, -0.313645443818603, -0.660016141198812, 1.07104977404073,
            0.559632480471393, -0.47740746012516, -1.26199769642326, -1.40486191330444, -0.418517018253652,
            0.284936091689505, -0.215531076378674, -0.200889269720281, 0.341231176941461, 0.0307350667648337,
            -0.212527052910377, -0.3019678509188, 0.54491723178866, -0.286914381308269, 0.370374100647823,
            -0.496258248067704, 0.624528657777646, 0.287884026214139, 0.022095151910937, 0.910293732936019,
            -0.13076488639207, 0.0857209529827562, -0.0922302696963889, 0.498136631416287, 0.937133644376614,
            0.395090607856869, -1.04727952099579, -0.54974694800345, 0.442372971174454, -0.745558450753062,
            -0.0920496108021893, 0.75549238586293, 0.562496351046743, 0.259183349320614, 0.405816113039412,
            -0.0969485695700491, -0.507388915258978, -0.474246597197329, -0.209335517183595, 0.187390427612498,
            -0.0522568530719332, 0.0806559868641646, -0.0397654339013217, -0.269582356665396, 0.791793553908743,
            0.344208857844796, -0.180165785909583, -0.7927695046551, 0.0311635012097026, -0.579429950080662,
            -0.264770995160963, 0.869512689697827, 0.765479119494175, -0.173588059680979, -0.199781736503338,
            -0.58712767650975, -0.457389854855, 0.3891865514653, 0.707309743580534, -0.121997864690072,
            0.0447174402649954, 0.0319336975869795, 0.0117988435665652, -0.593691059339064, -0.838107176656365,
            -0.247955128152877
        ])

        assert_array_almost_equal(cph.coef_, coef)

    @staticmethod
    def test_alpha(rossi):
        cph = CoxPHSurvivalAnalysis(alpha=-0.0001)

        with pytest.raises(ValueError, match=r"alpha must be positive, but was -0\.0001"):
            cph.fit(rossi.x.values, rossi.y)

        cph.set_params(alpha=-1.25)
        with pytest.raises(ValueError, match=r"alpha must be positive, but was -1\.25"):
            cph.fit(rossi.x.values, rossi.y)

    @staticmethod
    def test_alpha_array(rossi):
        cph = CoxPHSurvivalAnalysis(alpha=numpy.array([], dtype=float))

        with pytest.raises(ValueError,
                           match=r"Length alphas \(0\) must match number of features \(7\)"):
            cph.fit(rossi.x.values, rossi.y)

        alphas = numpy.ones(rossi.x.shape[1])
        alphas[-2] = -1e-4
        cph.set_params(alpha=alphas)
        with pytest.raises(ValueError, match=r"alpha must be positive, but was"):
            cph.fit(rossi.x.values, rossi.y)

        cph.set_params(alpha=alphas[:-2])
        with pytest.raises(ValueError,
                           match=r"Length alphas \(5\) must match number of features \(7\)"):
            cph.fit(rossi.x.values, rossi.y)

    @staticmethod
    def test_ties(rossi):
        cph = CoxPHSurvivalAnalysis(ties="xyz")
        with pytest.raises(ValueError, match="ties must be one of 'breslow', 'efron'"):
            cph.fit(rossi.x.values, rossi.y)

    @staticmethod
    def test_convergence(rossi):
        cph = CoxPHSurvivalAnalysis(n_iter=1)

        with pytest.warns(ConvergenceWarning,
                          match="Optimization did not converge: Maximum number of iterations has been exceeded."):
            cph.fit(rossi.x.values, rossi.y)

    @staticmethod
    def test_verbose(rossi):
        cph = CoxPHSurvivalAnalysis(verbose=99)
        cph.fit(rossi.x.values, rossi.y)

        cph.set_params(n_iter=1)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore")
            cph.fit(rossi.x.values, rossi.y)

    @staticmethod
    def test_cum_baseline_hazard(rossi):
        cph = CoxPHSurvivalAnalysis()
        cph.fit(rossi.x.values, rossi.y)

        expected_x = numpy.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52])
        assert_array_almost_equal(cph.cum_baseline_hazard_.x, expected_x)

        expected_y = numpy.array(
            [0.00678640369024364, 0.0135929334270716, 0.0204043079886091, 0.0272294776707967, 0.0340761479284598,
             0.0409513630548852, 0.0478644598407522, 0.0824874121212009, 0.096533196335404, 0.103576022547612,
             0.117797217724239, 0.13229957774496, 0.139567689198792, 0.161464121391667, 0.176251599922103,
             0.191124391501441, 0.213510322480229, 0.236290954886606, 0.251649773106939, 0.290325398473108,
             0.305965134135433, 0.313818453028679, 0.321716523315376, 0.35335069072333, 0.377266607384033,
             0.401708829897942, 0.418104688092493, 0.434591703603588, 0.45124335888492, 0.459626329898386,
             0.476473421812951, 0.493441590730406, 0.510649810315838, 0.54536240502959, 0.571724727186497,
             0.607219385133454, 0.616212515733231, 0.634272429676232, 0.670563043622984, 0.689028239653618,
             0.72608698374096, 0.744888154417096, 0.763829951751727, 0.802133842428817, 0.811813515937835,
             0.831261170527727, 0.880363253205648, 0.910240767958261, 0.950727380604515])

        actual_y = [cph.cum_baseline_hazard_(v) for v in expected_x]
        # check that values increase
        assert (numpy.diff(actual_y) > 0).all()
        assert_array_almost_equal(actual_y, expected_y)

    @staticmethod
    def test_predict_cumulative_hazard_function(rossi):
        cph = CoxPHSurvivalAnalysis()
        xc = standardize(rossi.x, with_std=False)
        cph.fit(xc, rossi.y)

        test_idx = [9, 3, 313, 122, 431]
        f = cph.predict_cumulative_hazard_function(xc.values[test_idx, :])
        assert len(f) == len(test_idx)

        expected_x = numpy.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52])
        assert_array_almost_equal(f[0].x, expected_x)

        expected_y = numpy.array([
            [0.00254216942097877, 0.00509187800740175, 0.00764340145273802, 0.010200092514897, 0.0127648376374983,
             0.0153402755947446, 0.0179299039244832, 0.0308995730701458, 0.0361610878212277, 0.0387993124614275,
             0.0441265357092526, 0.0495590825867252, 0.0522816985007711, 0.0604840458543902, 0.0660233973945142,
             0.0715947069840011, 0.0799804193276732, 0.0885139858140443, 0.0942673597370006, 0.108755149828895,
             0.114613754705306, 0.117555587825594, 0.120514184703119, 0.132364262699251, 0.141323103756249,
             0.150479097636328, 0.156620944074496, 0.1627969378097, 0.169034605180673, 0.172174844627085,
             0.178485722059738, 0.184841954627212, 0.191288109559384, 0.204291358530705, 0.214166616813619,
             0.227462824666768, 0.23083162832967, 0.237596825784311, 0.251191196714029, 0.258108211799492,
             0.271990322310271, 0.279033192636448, 0.286128741348309, 0.300477280578859, 0.30410326145516,
             0.311388302997746, 0.329781817263577, 0.340973857679346, 0.356140038962709],
            [0.000956575355883489, 0.00191598757220922, 0.00287608268924823, 0.00383812229309464, 0.00480319250366237,
             0.00577228624705267, 0.0067467195876016, 0.0116269867233485, 0.0136068057330369, 0.0145995250432739,
             0.0166040690489197, 0.0186482445550004, 0.0196727188742257, 0.0227591234521894, 0.0248434877463747,
             0.0269398773140096, 0.0300952790363686, 0.0333063157718797, 0.0354712130688363, 0.0409227234397632,
             0.0431272173646708, 0.0442341794108535, 0.045347449375349, 0.0498064332978417, 0.0531774936613566,
             0.0566227392976336, 0.0589338122315965, 0.0612577342158803, 0.0636048629461811, 0.0647864819371195,
             0.0671611583753846, 0.0695529011837649, 0.0719784802570701, 0.0768713829132565, 0.0805872755789716,
             0.0855904137074012, 0.0868580375471612, 0.0894036669255148, 0.094518998776647, 0.0971217545616966,
             0.10234535795047, 0.104995470933936, 0.107665405902953, 0.113064518529989, 0.114428913805359,
             0.117170151721572, 0.124091320039304, 0.12830269555009, 0.134009473052331],
            [0.00295687337931504, 0.00592251578772006, 0.00889026911287345, 0.0118640330478987, 0.0148471649018048,
             0.0178427339118856, 0.0208548081691458, 0.0359402187318172, 0.0420600440959181, 0.0451286421771019,
             0.051324895061419, 0.0576436530133163, 0.0608104091122242, 0.0703508049401599, 0.0767937905934741,
             0.0832739476109926, 0.0930276207497605, 0.102953267469424, 0.109645188965153, 0.126496371460824,
             0.133310690229679, 0.13673242442567, 0.140173657049699, 0.153956837619969, 0.16437713392766,
             0.175026744587659, 0.182170510098719, 0.189353993353509, 0.196609211060892, 0.200261717596116,
             0.207602088118528, 0.21499521255642, 0.222492928389435, 0.237617396652884, 0.249103605278338,
             0.264568822789954, 0.268487179209778, 0.27635598295447, 0.292168002868996, 0.300213390246307,
             0.316360088683985, 0.32455186206053, 0.332804907244929, 0.349494122893875, 0.353711609831828,
             0.362185059802043, 0.383579107042115, 0.396596904397598, 0.414237144002626],
            [0.000722773155807687, 0.00144768979833789, 0.00217312242980782, 0.00290002428464583, 0.00362921601781963,
             0.00436144786853125, 0.00509771423410398, 0.0087851666205712, 0.0102810864399135, 0.0110311693939413,
             0.0125457710277853, 0.0140903175943128, 0.0148643941291055, 0.0171964324397243, 0.0187713450166532,
             0.0203553436993314, 0.0227395151571085, 0.025165723547765, 0.02680148558334, 0.0309205602913373,
             0.0325862409105006, 0.0334226438625083, 0.0342638129774717, 0.0376329504547583, 0.0401800701587836,
             0.0427832430774134, 0.0445294531041597, 0.0462853716693838, 0.0480588248835527, 0.0489516374380368,
             0.0507459052630967, 0.0525530681665324, 0.0543857972146842, 0.0580827967998402, 0.0608904663181093,
             0.0646707580763963, 0.065628554529551, 0.0675519916827172, 0.0714170552371003, 0.0733836561964356,
             0.0773305280061372, 0.0793329113129253, 0.0813502717973751, 0.0854297556017638, 0.0864606710156758,
             0.0885319068752816, 0.0937614318009609, 0.0969434802924884, 0.101255430793218],
            [0.00139255604317491, 0.00278924190960753, 0.00418692192408016, 0.00558743266886364, 0.00699235528739893,
             0.00840313525414264, 0.00982168845928897, 0.0169262468721653, 0.0198084128294806, 0.0212535862454006,
             0.024171746170495, 0.0271476005418148, 0.0286390020247613, 0.033132104758687, 0.036166464860265,
             0.0392183310235561, 0.0438118778989689, 0.0484864166934194, 0.0516380145212199, 0.0595741730943763,
             0.0627834146020013, 0.0643948994447748, 0.0660155671812037, 0.0725068331013381, 0.0774143298837369,
             0.082429823541932, 0.0857942198339895, 0.0891773214194287, 0.0925942067462213, 0.0943143751118822,
             0.0977713636328095, 0.10125319690506, 0.104784288079377, 0.111907240934718, 0.11731673508027,
             0.124600165698393, 0.126445537552971, 0.130151394653205, 0.137598153801166, 0.141387173950061,
             0.14899155181893, 0.152849513272278, 0.156736331025966, 0.164596210407468, 0.16658245945132,
             0.170573078070673, 0.18064872418408, 0.186779528601685, 0.195087298030326]
        ])

        for i, ff in enumerate(f):
            actual_y = ff(expected_x)
            # check that values increase
            assert (numpy.diff(actual_y) > 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])

        pipe = make_pipeline(StandardScaler(with_std=False), CoxPHSurvivalAnalysis())
        pipe.fit(rossi.x, rossi.y)
        f = pipe.predict_cumulative_hazard_function(xc.values[test_idx, :])
        assert len(f) == len(test_idx)

        for i, ff in enumerate(f):
            actual_y = [ff(v) for v in expected_x]
            # check that values increase
            assert (numpy.diff(actual_y) > 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])

    @staticmethod
    def test_predict_survival_function(rossi):
        cph = CoxPHSurvivalAnalysis()
        xc = standardize(rossi.x, with_std=False)
        cph.fit(xc, rossi.y)

        test_idx = [9, 3, 313, 122, 431]
        f = cph.predict_survival_function(xc.values[test_idx, :])
        assert len(f) == len(test_idx)

        expected_x = numpy.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30,
             31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52])
        assert_array_almost_equal(f[0].x, expected_x)

        expected_y = numpy.array([
            [0.997461059155262, 0.994921063628358, 0.992385735058868, 0.989851752006058, 0.987316287353143,
             0.984776787077271, 0.982229880405155, 0.969572939420418, 0.964484914195169, 0.961943739897691,
             0.956832876286024, 0.95164893059497, 0.949061479987368, 0.941308786676055, 0.936108961524705,
             0.930908109920408, 0.923134421802298, 0.915290309566414, 0.910039421812024, 0.896950008976804,
             0.891710496383746, 0.889091087732452, 0.886464513020254, 0.876021834875649, 0.868208744867335,
             0.860295712033317, 0.855028100866882, 0.849763725779484, 0.844479679508505, 0.841831970503126,
             0.836536000803172, 0.831235646418842, 0.825894605739379, 0.815224813877833, 0.807213878413131,
             0.796552033034344, 0.793873120526445, 0.788520538165122, 0.777873630457932, 0.772511632709885,
             0.761861634114672, 0.756514792077836, 0.751165903610347, 0.740464726897091, 0.737784677797559,
             0.732429416106706, 0.719080607292328, 0.71107749717937, 0.700374533644241],
            [0.999043882016474, 0.998085846760273, 0.997128049274331, 0.996169233883962, 0.995208324378805,
             0.994244341388665, 0.993275988428092, 0.988440345477267, 0.986485348400118, 0.985506531271212,
             0.983533018718637, 0.981524558134704, 0.980519526333744, 0.977497911740073, 0.975462571926908,
             0.973419764375874, 0.970353074837982, 0.967242232655452, 0.965150517551612, 0.95990330510734,
             0.957789534855612, 0.9567298847967, 0.955665378803038, 0.951413568664925, 0.948211695979523,
             0.944950494817014, 0.942769166879369, 0.940580788681338, 0.938375713299261, 0.937267565568928,
             0.935044498941264, 0.932810785284511, 0.930550920812338, 0.926008946520639, 0.922574381856697,
             0.917970142218167, 0.916807238599598, 0.914476355200083, 0.909810449176314, 0.907445513776142,
             0.902717737080621, 0.900328600224742, 0.897927987577297, 0.893093037207259, 0.891875336188489,
             0.88943384158804, 0.883299174284255, 0.879587091746084, 0.874581779605126],
            [0.997047493865252, 0.994094987736893, 0.991149132479536, 0.988206067094847, 0.985262510789073,
             0.982315505128104, 0.979361149485527, 0.964697962626958, 0.958812207831876, 0.955874508145704,
             0.949969979834156, 0.943986274173002, 0.941001628199154, 0.932066788913148, 0.926080800490561,
             0.920099053753847, 0.91116833279606, 0.902169133142995, 0.896152043528463, 0.881177346924239,
             0.875193135907483, 0.872203575284039, 0.869207278320275, 0.857309023710775, 0.848421992818626,
             0.839434570137547, 0.833459215086135, 0.827493527632619, 0.821511608241697, 0.818516504870906,
             0.812530287850531, 0.806545301458209, 0.800520667700624, 0.788504317779599, 0.779499208968453,
             0.76753682302423, 0.764535224692751, 0.758542854366408, 0.746643087255005, 0.740660154164664,
             0.728796971370325, 0.722851218124105, 0.716910044469078, 0.705044665474413, 0.702077410385152,
             0.696153525886713, 0.681418170940623, 0.67260509515153, 0.660844217439567],
            [0.999277487981792, 0.998553357599042, 0.997829237091254, 0.99710417672379, 0.996377361626995,
             0.995648049432879, 0.994915257060425, 0.991253310198176, 0.989771583274112, 0.989029450846354,
             0.9875325990774, 0.986008486326716, 0.985245535623559, 0.982950582288741, 0.981403739445045,
             0.979850427761197, 0.977517078995485, 0.9751482936019, 0.973554486943764, 0.969552590992207,
             0.967938970293575, 0.967129721757568, 0.966316544163349, 0.963066369097215, 0.960616445221447,
             0.958119046496724, 0.956447429333005, 0.954769459155805, 0.953077720742534, 0.952227180730954,
             0.950520162020759, 0.948803968432749, 0.947066660332597, 0.943571819475399, 0.9409262972535,
             0.937376036066338, 0.936478650448705, 0.934679123812276, 0.931073502063006, 0.929244251313332,
             0.925583871649869, 0.923732352300608, 0.921870729568298, 0.918117633409758, 0.917171619503635,
             0.915273906739705, 0.910499952666305, 0.907607302382901, 0.903702170040254],
            [0.998608413113071, 0.997214644411471, 0.995821831013303, 0.994428148000897, 0.993032034348724,
             0.991632072399662, 0.99022638680066, 0.983216197230745, 0.980386484787682, 0.978970679590968,
             0.976118050823751, 0.97321758347997, 0.971767207148281, 0.967410751583306, 0.964479728136077,
             0.961540752050541, 0.957133998596443, 0.952670279630427, 0.949672572311715, 0.942165647657932,
             0.93914685715923, 0.937634655006455, 0.936116291486253, 0.930059391448493, 0.925506309233532,
             0.920876059409491, 0.917783073342697, 0.91468336624063, 0.911563331503877, 0.909996636973231,
             0.906856220364748, 0.903704188812084, 0.900518754266508, 0.89412719205641, 0.889303474943224,
             0.882849825667986, 0.881222141750833, 0.877962502029932, 0.871448809785976, 0.868153120346739,
             0.861576393621591, 0.85825886866239, 0.854929447472668, 0.848236143873358, 0.846553007760733,
             0.84318146930209, 0.834728527145727, 0.82962662516592, 0.822762829346384]
        ])

        for i, ff in enumerate(f):
            actual_y = [ff(v) for v in expected_x]
            # check that values decrease
            assert (numpy.diff(actual_y) < 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])

        pipe = make_pipeline(StandardScaler(with_std=False), CoxPHSurvivalAnalysis())
        pipe.fit(rossi.x, rossi.y)
        f = pipe.predict_survival_function(xc.values[test_idx, :])
        assert len(f) == len(test_idx)

        for i, ff in enumerate(f):
            actual_y = ff(expected_x)
            # check that values decrease
            assert (numpy.diff(actual_y) < 0).all()
            assert_array_almost_equal(actual_y, expected_y[i, :])
