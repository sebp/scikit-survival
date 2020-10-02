from os.path import join, dirname

import numpy
from numpy.testing import assert_array_almost_equal
import pandas
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline

from sksurv.datasets import load_breast_cancer, get_x_y
from sksurv.linear_model.coxnet import CoxnetSurvivalAnalysis
from sksurv.preprocessing import OneHotEncoder
from sksurv.util import Surv

BREAST_CANCER_COEFFICIENTS_FILE = join(dirname(__file__), 'data', 'breast_cancer_glmnet_coefficients.csv')
BREAST_CANCER_HIGH_COEFFICIENTS_FILE = join(dirname(__file__), 'data', 'breast_cancer_glmnet_coefficients_high.csv')
EXAMPLE_FILE = join(dirname(__file__), 'data', 'cox-example.csv')
EXAMPLE_COEF_FILE = join(dirname(__file__), 'data', 'cox-example-coef-{}.csv')
SIMPLE_COEF_FILE = join(dirname(__file__), 'data', 'cox-simple-coef.csv')


@pytest.fixture(params=[0, -1, -1e-6, -numpy.infty])
def invalid_positive_int(request):
    return request.param


@pytest.fixture(params=[(-1, 1), (1, -1e-7), (-1e-6, 1)])
def negative_float_array(request):
    a, b = request.param
    penalty = a * numpy.ones(30, dtype=float)
    penalty[11] = b
    return penalty


@pytest.fixture(params=[-numpy.infty, numpy.infty, numpy.nan])
def infinite_float_array(request):
    penalty = numpy.zeros(30)
    penalty[11] = request.param
    return penalty


@pytest.fixture
def make_example_coef():
    def _make_example_coef(kind):
        return pandas.read_csv(EXAMPLE_COEF_FILE.format(kind))

    return _make_example_coef


@pytest.fixture(params=[False, True])
def normalize_options(request):
    return request.param


def assert_columns_almost_equal(actual, expected, decimal=6):
    for i, col in enumerate(expected.columns):
        assert_array_almost_equal(expected.loc[:, col].values, actual.loc[:, col].values,
                                  decimal=decimal,
                                  err_msg="Column %d: %s" % (i, col))


def assert_predictions_equal(coxnet, x, expected_pred):
    pred = numpy.array([
        coxnet.predict(x.iloc[122:123, :], alpha=a)[0] for a in coxnet.alphas_])
    assert_array_almost_equal(pred, expected_pred)

    pred_last = coxnet.predict(x.iloc[122:123, :])
    assert_array_almost_equal(pred_last, expected_pred[-1])


class TestCoxnetSurvivalAnalysis(object):
    def _fit_example(self, **kwargs):
        x, y = get_x_y(pandas.read_csv(EXAMPLE_FILE), ["status", "time"],
                       pos_label=1)
        coxnet = CoxnetSurvivalAnalysis(**kwargs)
        coxnet.fit(x.values, y)

        return x, coxnet

    def test_example_1(self, make_example_coef):
        expected_alphas = numpy.array(
            [0.468899812444165, 0.427244045448662, 0.389288861984934, 0.354705512411255, 0.323194452297996,
             0.294482747917078, 0.268321712220587, 0.244484750832538, 0.222765399396784, 0.202975535281541,
             0.184943748151146, 0.168513855291727, 0.153543548831257, 0.139903163136841, 0.127474551713039,
             0.116150063873439, 0.105831612321913, 0.09642982356738, 0.0878632638133844, 0.0800577336175056,
             0.0729456252112861, 0.0664653369140829, 0.0605607395687804, 0.0551806903778823, 0.0502785899290667,
             0.0458119785733689, 0.0417421686600145, 0.0380339094424964, 0.034655081753472, 0.0315764198039011,
             0.0287712576967823, 0.0262152984599091, 0.0238864035971209, 0.0217644013352397, 0.0198309119058206,
             0.018069188348391, 0.0164639714562898, 0.0150013576087198, 0.0136686783442352, 0.0124543906325894,
             0.0113479768945291, 0.010339853903554, 0.00942128978059378, 0.00858432836264852, 0.00782172029031119,
             0.00712686021728439, 0.00649372959803068, 0.00591684455801036, 0.00539120839498366, 0.00491226829996627,
             0.00447587592297602])

        x, coxnet = self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.5)
        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef(1)
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_dev = numpy.array([
            0, 0.00149456666347458, 0.00448476987407723, 0.00808771904805364, 0.0113599013712407, 0.0152010020968126,
            0.018627406781334, 0.0216589556362678, 0.0245392698779375, 0.0276857676108435, 0.0307773347985121,
            0.0340774598891008, 0.0369400175894911, 0.0394113544606718, 0.0417585954278998, 0.0440721909385255,
            0.0460451436077026, 0.0477767269023748, 0.0492704173774208, 0.0505552133421733, 0.0516577234368319,
            0.0526016991057607, 0.0534152945890704, 0.054170718154795, 0.0548231528901663, 0.0553769276506753,
            0.0558516430053506, 0.0562640599593089, 0.0566166295700916, 0.0569206732927765, 0.0571998906771315,
            0.0574259968668313, 0.0576288674073685, 0.0578011825021085, 0.0579483305146174, 0.0580721211069517,
            0.0581758085541296, 0.058263711446059, 0.0583380031206903, 0.0584007416302968, 0.0584532957120563,
            0.0584981837358404, 0.0585328099028761, 0.0585655649604582, 0.0585936711622123, 0.0586179532680806,
            0.0586382158975456, 0.058655105808516, 0.0586691773302103, 0.0586810916901528, 0.0586884909745359
        ])
        assert_array_almost_equal(coxnet.deviance_ratio_, expected_dev)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([
            0, 0.00140948454488767, 0.00160298997140239, -0.00830742573928443, -0.0174234349495347, -0.0200048263613719,
            -0.0224801816347042, -0.0248329460065182, -0.0216924070471563, -0.0325287602403503, -0.0510221489452056,
            -0.0825622454695751, -0.112204332292886, -0.139972771206084, -0.165186968563414, -0.18482492543353,
            -0.202718548788946, -0.219680397975635, -0.235597592089928, -0.250505411809673, -0.264444425661016,
            -0.277456457500562, -0.288750431878708, -0.297632304112204, -0.305835046675932, -0.313485105192284,
            -0.320669158625457, -0.327488555939263, -0.333453200575523, -0.338454390304256, -0.343545582912776,
            -0.348034928885305, -0.354378838938737, -0.360720166847284, -0.366107613620401, -0.370999342778936,
            -0.375502769096629, -0.379949069210773, -0.384261695764238, -0.388813847771416, -0.393135194649681,
            -0.397093422412222, -0.40046816729172, -0.402004566839427, -0.40331328958967, -0.404500835078198,
            -0.405588873666053, -0.406584748259676, -0.407495670627834, -0.407927461818659, -0.408063340186696
        ]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    def test_example_1_penalty_factor_1(self, make_example_coef):
        expected_alphas = numpy.array(
            [1.58712986523032, 1.44613362231646, 1.31766309702114, 1.20060553911345, 1.09394705202614, 0.99676381096855,
             0.908214061198282, 0.827530827144269, 0.754015269231302, 0.687030630865956, 0.625996723155588,
             0.57038489958971, 0.519713477156817, 0.47354356423657, 0.431475259130545, 0.393144186305032,
             0.358218340344475, 0.326395210279388, 0.297399159381062, 0.270979037728078, 0.246906006865839,
             0.224971557717347, 0.204985704577333, 0.186775339546958, 0.170182733156013, 0.1550641681857,
             0.141288694858838, 0.128736996615539, 0.117300356650245, 0.106879716258767, 0.0973848168391796,
             0.088733418114989, 0.0808505858092015, 0.0736680425994639, 0.0671235767325608, 0.0611605031759696,
             0.0557271726392266, 0.0507765242124814, 0.0462656777474094, 0.0421555624498806, 0.0384105784674319,
             0.0349982885403755, 0.0318891370457737, 0.0290561940007717, 0.0264749218079695, 0.0241229627224915,
             0.0219799451998966, 0.0200273074476052, 0.0182481366515322, 0.0166270224853822, 0.015149923633775,
             0.0138040461730896, 0.0125777327566178, 0.011460361644204, 0.0104422547018131, 0.00951459357416398,
             0.00866934330435397, 0.00789918273890653, 0.00719744111544197, 0.00655804028372505, 0.00597544205963532,
             0.00544460025606573, 0.00496091697526395, 0.00452020278404162])

        pf = numpy.ones(30)
        pf[4] = 0.125
        x, coxnet = self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.5, penalty_factor=pf)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("1-pf")
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([
            0, -0.0219470160430587, -0.0423872317402525, -0.0613777857697107, -0.0789932877972767, -0.0953091589085253,
            -0.110400665798654, -0.124342123109189, -0.137206242394496, -0.149063613776774, -0.159982305706598,
            -0.170027568241602, -0.179261625730142, -0.187743545598723, -0.195047607087029, -0.200036792234896,
            -0.220238913103036, -0.239384552773407, -0.253379565373219, -0.264180012747967, -0.274257453314554,
            -0.281696215417802, -0.278053682674572, -0.276239842910816, -0.288084434992464, -0.30234982200802,
            -0.315735609513988, -0.32825728518968, -0.335338196765289, -0.341667043049741, -0.347724135480807,
            -0.353500260626603, -0.358992026119048, -0.364198632953538, -0.369121567561978, -0.371981666745392,
            -0.374171559462367, -0.376100523422468, -0.377881839851226, -0.379598485711978, -0.38134957880695,
            -0.383108070287709, -0.385083683420684, -0.386218437874565, -0.387068637979054, -0.388095519736864,
            -0.391347677297057, -0.394371697177822, -0.396812702028453, -0.399074022881495, -0.401165142344145,
            -0.403587140384981, -0.405974801639294, -0.408793015715854, -0.411377351836687, -0.413496980133616,
            -0.414144504730355, -0.41438652016618, -0.414598984650713, -0.414796366581722, -0.414979630970168,
            -0.415149243472181, -0.415305936592074, -0.414884495552163
        ]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    def test_example_1_penalty_factor_2(self, make_example_coef):
        expected_alphas = numpy.array(
            [1.58712986523032, 1.44613362231646, 1.31766309702114, 1.20060553911345, 1.09394705202614, 0.99676381096855,
             0.908214061198282, 0.827530827144269, 0.754015269231302, 0.687030630865956, 0.625996723155588,
             0.57038489958971, 0.519713477156817, 0.47354356423657, 0.431475259130545, 0.393144186305032,
             0.358218340344475, 0.326395210279388, 0.297399159381062, 0.270979037728078, 0.246906006865839,
             0.224971557717347, 0.204985704577333, 0.186775339546958, 0.170182733156013, 0.1550641681857,
             0.141288694858838, 0.128736996615539, 0.117300356650245, 0.106879716258767, 0.0973848168391796,
             0.088733418114989, 0.0808505858092015, 0.0736680425994639, 0.0671235767325608, 0.0611605031759696,
             0.0557271726392266, 0.0507765242124814, 0.0462656777474094, 0.0421555624498806, 0.0384105784674319,
             0.0349982885403755, 0.0318891370457737, 0.0290561940007717, 0.0264749218079695, 0.0241229627224915,
             0.0219799451998966, 0.0200273074476052, 0.0182481366515322, 0.0166270224853822, 0.015149923633775,
             0.0138040461730896, 0.0125777327566178, 0.011460361644204, 0.0104422547018131, 0.00951459357416398,
             0.00866934330435397, 0.00789918273890653, 0.00719744111544197, 0.00655804028372505, 0.00597544205963532,
             0.00544460025606573, 0.00496091697526395, 0.00452020278404162])

        pf = numpy.ones(30)
        pf[4] = 0.125
        pf[10] = 1.25
        pf[12] = 0.75
        x, coxnet = self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.5, penalty_factor=pf)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("1-pf2")
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([
            0, -0.0219470160430587, -0.0423872317402525, -0.0613777857697107, -0.0789932877972767, -0.0953091589085253,
            -0.110400665798654, -0.124342123109189, -0.137206242394496, -0.149063613776774, -0.159982305706598,
            -0.170027568241602, -0.179261625730142, -0.187743545598723, -0.195047607087029, -0.200036792234896,
            -0.220238913103036, -0.239384552773407, -0.253379565373219, -0.264180012747967, -0.274257453314554,
            -0.281696215417802, -0.278053682674572, -0.276239842910816, -0.288084434992464, -0.30234982200802,
            -0.315735609513988, -0.32825728518968, -0.335338196765289, -0.341667043049741, -0.347724135480807,
            -0.352444163392331, -0.355262839986677, -0.358048733783623, -0.360779034461829, -0.363435553170601,
            -0.366294732280829, -0.368855198722435, -0.371238650310903, -0.373510228445821, -0.375780785361733,
            -0.377992809262836, -0.380025554130693, -0.381590783714262, -0.382836176775286, -0.38422559085889,
            -0.385551299387758, -0.386727466180965, -0.389275888238276, -0.392185115768271, -0.39486863153171,
            -0.397870315511183, -0.401029223528979, -0.404277261469497, -0.407255492228115, -0.409734816795972,
            -0.410650182963323, -0.411197059249143, -0.411692846517166, -0.412145302413102, -0.412561364082062,
            -0.4129435341443, -0.41329427736124, -0.413095216567575
        ]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    def test_example_1_unpenalized(self, make_example_coef):
        expected_alphas = numpy.array(
            [0.486489606631737, 0.443271210800114, 0.403892218139282, 0.368011546653386, 0.335318415131023,
             0.305529651307061, 0.278387239159946, 0.253656084100358, 0.231121976694345, 0.210589737283679,
             0.191881525433034, 0.174835299561207, 0.159303413414467, 0.145151337225331, 0.132256492480081,
             0.120507190202318, 0.109801663556478, 0.100047186392186, 0.0911592700947043, 0.0830609317849671,
             0.0756820275307113, 0.0689586447932912, 0.0628325488478443, 0.0572506783819629, 0.0521646859039908,
             0.047530518980179, 0.0433080386735795, 0.0394606718797749, 0.0359550945481447, 0.0327609430448805,
             0.0298505511577159, 0.0271987104644308, 0.024782451989558, 0.0225808472581054, 0.0205748270231182,
             0.0187470160969839, 0.0170815828558693, 0.0155641021137689, 0.0141814301784446, 0.012921591007051,
             0.0117736724753817, 0.0107277318622698, 0.00977470973049374, 0.00890635145826557, 0.00811513574164646,
             0.0073942094486099, 0.00673732826049059, 0.00613880258668338, 0.00559344828413133, 0.00509654175475855,
             0.00464377903192347])

        pf = numpy.ones(30)
        pf[0] = 0
        pf[29] = 0
        x, coxnet = self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.5, penalty_factor=pf)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("1-unpen")
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([
            0.0373756105322525, 0.0389973815599095, 0.0405683877195706, 0.0280831344913328, 0.0159083690720834,
            0.0103802188902682, 0.00515467167201751, 0.000240537790719531, -0.00398308401428462, 0.00430672528959097,
            -0.0108117367591171, -0.0418650923281516, -0.0753864542967495, -0.106813391745568, -0.135786421270196,
            -0.159991704303222, -0.180394076619709, -0.199534421788881, -0.217451880596263, -0.234197183369495,
            -0.249822344205612, -0.264380049409194, -0.277923294024265, -0.290539010996487, -0.299717367878573,
            -0.30821456747717, -0.316104156500028, -0.323418551099414, -0.330190332294807, -0.33623057935319,
            -0.342359036083179, -0.347196305835025, -0.353818779401235, -0.360222871635933, -0.366122922710483,
            -0.371515131203764, -0.375983474095953, -0.380333208342917, -0.38462151635402, -0.388555036993323,
            -0.392161768186302, -0.395575701614631, -0.397949529352037, -0.399599074245129, -0.401120250339891,
            -0.402505065734498, -0.403764001966903, -0.404916031377837, -0.405970148147989, -0.406934334318935,
            -0.407120115909973
        ]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    def test_example_2(self, make_example_coef):
        expected_alphas = numpy.array(
            [0.260499895802314, 0.237357803027034, 0.21627158999163, 0.197058618006253, 0.179552473498887,
             0.163601526620599, 0.149067617900326, 0.135824861573632, 0.123758555220435, 0.112764186267523,
             0.102746526750637, 0.0936188084954038, 0.0853019715729203, 0.0777239795204672, 0.0708191953961327,
             0.0645278132630217, 0.0587953401788405, 0.0535721242041, 0.0488129243407691, 0.044476518676392,
             0.0405253473396034, 0.0369251871744905, 0.0336448553159891, 0.0306559390988235, 0.0279325499605926,
             0.0254510992074272, 0.0231900937000081, 0.0211299496902758, 0.0192528231963733, 0.0175424554466117,
             0.0159840320537679, 0.0145640546999495, 0.0132702242206227, 0.0120913340751331, 0.0110171732810114,
             0.0100384379713283, 0.00914665080904989, 0.00833408756039986, 0.00759371019124175, 0.00691910590699411,
             0.00630443160807172, 0.00574436327975223, 0.00523404987810766, 0.00476907131258251, 0.004345400161284,
             0.00395936678738022, 0.00360762755446149, 0.00328713586556131, 0.00299511577499092])

        x, coxnet = self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.9)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef(2)
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_dev = numpy.array([
            0, 0.00196732895383744, 0.00576766085750443, 0.0102026366034422, 0.0141277450441783, 0.0184873761337667,
            0.0221651775163659, 0.0252330304992086, 0.028286869539667, 0.0314543603735154, 0.0347699039782729,
            0.037935769673122, 0.0406135045641876, 0.0429430816958846, 0.0451503007038936, 0.0472013548042505,
            0.0489282107777369, 0.0503557288828739, 0.0515776800400246, 0.0526038591451334, 0.0534644227458316,
            0.0541855761749597, 0.0547999582318951, 0.0553819071844805, 0.0558724141507787, 0.0562822436073126,
            0.056629843616449, 0.0569313013655897, 0.0571888033859859, 0.0574127557793992, 0.0576185678698586,
            0.0577939984518394, 0.0579432777114924, 0.0580676788843513, 0.0581729295908792, 0.058261762526089,
            0.0583356961430689, 0.0583985935302897, 0.0584514361607712, 0.0584922650630193, 0.058529400715956,
            0.05856227872849, 0.0585907155451063, 0.0586148494867005, 0.0586353364420467, 0.0586528427291789,
            0.0586641086168485, 0.0586788544493303, 0.0586869714887646
        ])
        assert_array_almost_equal(coxnet.deviance_ratio_, expected_dev)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([
            0, 0.00196016970842094, 0.00140611753838389, -0.0116396716112419, -0.022089013534796, -0.0250863119930773,
            -0.0278225590136988, -0.0302883122217436, -0.0221472435312749, -0.0381375326325586, -0.0662858762239503,
            -0.101379025086768, -0.13387188966317, -0.16343807084084, -0.187948398579459, -0.207667002292175,
            -0.225935451658892, -0.242648974678585, -0.258268196953873, -0.272693412762738, -0.285997926279684,
            -0.298257592172903, -0.309383802664966, -0.317278273283531, -0.324463328947265, -0.331084072762535,
            -0.337239118102596, -0.343049267695076, -0.347880298318, -0.352715098645548, -0.356825310385595,
            -0.3619252765108, -0.367818211838333, -0.373221800261174, -0.377865968340762, -0.381894769178121,
            -0.385584019971888, -0.389340460428481, -0.392932674551871, -0.396271896579473, -0.399949396718947,
            -0.403353040387268, -0.405235426737329, -0.406286722398451, -0.407239920728446, -0.408098144214668,
            -0.408543625518507, -0.409505822297035, -0.409923859178006
        ]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    def test_example_2_normalize(self, make_example_coef):
        expected_alphas = numpy.array(
            [0.00831887750878913, 0.00757985135869546, 0.0069064782549347, 0.00629292576181799, 0.00573387958116825,
             0.00522449752241159, 0.00476036756183909, 0.00433746962776807, 0.00395214078060438, 0.00360104348621191,
             0.00328113670778857, 0.00298964956586034, 0.00272405733824878, 0.00248205959213538, 0.00226156025881296,
             0.00206064947854128, 0.00187758705825562, 0.00171078739884696, 0.00155880576146105, 0.00142032575386155,
             0.0012941479284703, 0.00117917939332615, 0.00107442434597765, 0.000978975448318574, 0.000892005967659349,
             0.000812762615963956, 0.000740559025229353, 0.000674769802494178, 0.000614825112984099,
             0.000560205744475609, 0.000510438610128125, 0.000465092650832102, 0.000423775101581237,
             0.000386128089529882, 0.000351825534269655, 0.000320570323476944, 0.000292091739468086,
             0.000266143114372309, 0.000242499693612633, 0.000220956689189248, 0.000201327505903829,
             0.00018344212516119, 0.000167145632349552, 0.000152296875044284, 0.000138767240413126,
             0.000126439541233365, 0.000115207000872171, 0.000104972328438487, 9.5646876095872e-05,
             8.71498712373515e-05, 7.94077168717352e-05, 7.23533541616886e-05, 6.59256815921643e-05,
             6.00690257383086e-05, 5.47326590488896e-05])

        x, coxnet = self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.9, normalize=True)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("2-norm")
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([
            0, 0.000639408747219786, -0.00052949581811348, -0.00655544513119031, -0.0126379123842854,
            -0.0154908462985011, -0.0173929193691804, -0.019286506654261, -0.0211630424740022, -0.0298515402290131,
            -0.0415450906301397, -0.0561550947027854, -0.0794886384588791, -0.102298331006247, -0.124312860420833,
            -0.145023690076741, -0.161527852664137, -0.177495378416267, -0.192896322287149, -0.207713020484007,
            -0.221931814547868, -0.235542897596703, -0.248302871817807, -0.258066101598933, -0.267578020734177,
            -0.276668591810926, -0.285326398847612, -0.293735142605935, -0.301713974346824, -0.308695984478683,
            -0.314247838968584, -0.319176436422011, -0.325263490815563, -0.331441729472062, -0.338538969805117,
            -0.344920772188639, -0.351102580604316, -0.356876599562591, -0.363158677531647, -0.369352370626939,
            -0.375075124027788, -0.380357592110288, -0.385230301184507, -0.3897221161163, -0.393859049671187,
            -0.395986139303465, -0.397770798375561, -0.399167571117783, -0.400486682367687, -0.40143075878309,
            -0.402309388629067, -0.403119489802433, -0.403943664491923, -0.404691055430968, -0.405377242302289
        ]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    @staticmethod
    def test_example_2_standardize(make_example_coef):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        x, y = get_x_y(pandas.read_csv(EXAMPLE_FILE), ["status", "time"],
                       pos_label=1)
        expected_alphas = numpy.array(
            [0.263066005037211, 0.239695946189997, 0.218402018960187, 0.198999785536952, 0.18132119305624,
             0.165213118007272, 0.15053603994994, 0.137162833055499, 0.124977665003457, 0.113874993697428,
             0.103758653109983, 0.0945410203385227, 0.0861422566576188, 0.0784896159941638, 0.0715168148356887,
             0.0651634581142872, 0.0593745160934302, 0.0540998477267133, 0.0492937663601004, 0.0449146440159821,
             0.0409245508315483, 0.037288926528462, 0.0339762810682614, 0.0309579219007116, 0.0282077054426604,
             0.0257018106348284, 0.0234185326151886, 0.0213380947218357, 0.0194424771970012, 0.0177152611085322,
             0.0161414861369557, 0.0147075209963485, 0.0134009453666594, 0.0122104423148384, 0.0111257002729774,
             0.0101373237244409, 0.00923675182439653, 0.00841618424987191, 0.00766851363708906, 0.00698726402087929,
             0.00636653474297097, 0.00580094934331044, 0.00528560899173708, 0.00481605005665997, 0.00438820544321646,
             0.0039983693660421, 0.00364316525153066, 0.00331951649156886, 0.0030246197954287])

        scaler = StandardScaler()
        coxnet = CoxnetSurvivalAnalysis(alpha_min_ratio=0.0001, l1_ratio=0.9)
        pipe = Pipeline([("standardize", scaler),
                         ("coxnet", coxnet)])
        pipe.fit(x.values, y)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("2-std")
        # rescale coefficients
        coef = pandas.DataFrame(coxnet.coef_ / scaler.scale_[:, numpy.newaxis],
                                columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef, 5)

    def test_example_2_with_alpha(self, make_example_coef):
        expected_alphas = numpy.array([0.45, 0.4, 0.35, 0.25, 0.1, 0.05, 0.001])

        x, coxnet = self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.9, alphas=expected_alphas, normalize=True)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("2-alpha")
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([0, 0, 0, 0, 0, 0, -0.25599344616167]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    def test_example_2_with_n_alpha_and_norm(self, make_example_coef):
        expected_alphas = numpy.array([
            0.00831887750878913, 0.00416931520551416, 0.00208960755397176, 0.00104728462934176, 0.00052488568620016,
            0.000263066005037211, 0.000131845323325978, 6.607919286444e-05, 3.31180478720517e-05, 1.65983427961291e-05,
            8.31887750878913e-06])

        x, coxnet = self._fit_example(l1_ratio=0.9, n_alphas=11,
                                      alpha_min_ratio=0.001,
                                      normalize=True)

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("2-nalpha-norm")
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_offset = numpy.dot(numpy.mean(x.values, axis=0), expected_coef)
        assert_array_almost_equal(coxnet.offset_, expected_offset)

        expected_pred = numpy.array([
            0, -0.0200793716023623, -0.142080657028422, -0.251017077112929, -0.31287130943375, -0.357757777608831,
            -0.394929617816296, -0.404002085451039, -0.408212946743633, -0.410651291215939, -0.411910901429096
        ]) - expected_offset
        assert_predictions_equal(coxnet, x, expected_pred)

    def test_example_2_with_n_alpha_interpolation(self, make_example_coef):
        x, coxnet = self._fit_example(l1_ratio=0.9, n_alphas=11,
                                      alpha_min_ratio=0.001)

        expected_alphas = numpy.array(
            [0.260499895802314, 0.130559222137355, 0.0654346153675493, 0.0327949938595266, 0.0164364322492795,
             0.00823773000971849, 0.00412864511493917, 0.00206922422378511, 0.0010370687644734, 0.000519765625147677,
             0.000260499895802314])
        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_coef = make_example_coef("2-nalpha")
        coef = pandas.DataFrame(coxnet.coef_, columns=expected_coef.columns, dtype=float)
        assert_columns_almost_equal(coef, expected_coef)

        expected_offset = numpy.array(
            [0, 0.000691134372356494, 0.00398228309120452, 0.00727343181005256, 0.0120190989555886, 0.0148545910591148]
        )

        expected_pred = numpy.array([
            [0, -0.00238292430438636, -0.0137302955033516, -0.0250776667023169, -0.11170732289777, -0.178968215018837],
            [0, 0.019393427730732, 0.111744000040494, 0.204094572350256, 0.202194208945816, 0.1712643455565],
            [0, -0.0142582806836494, -0.0821555291520897, -0.15005277762053, -0.227574637069682, -0.269396998188837],
            [0, -1.68697383231657e-05, -9.72026227669121e-05, -0.000177535507210586, 0.0889321572547518,
             0.161856892574037]
        ]) - expected_offset[numpy.newaxis]
        pred = numpy.column_stack([
            coxnet.predict(x.iloc[[122, 10, 22, 200], :], alpha=a) for a in [0.75, 0.25, 0.2, 0.15, 0.1, 0.075]])
        assert_array_almost_equal(pred, expected_pred)

    def test_example_2_predict_func(self):
        x, coxnet = self._fit_example(l1_ratio=0.9, n_alphas=11,
                                      alpha_min_ratio=0.001,
                                      fit_baseline_model=True)

        xtest = x.iloc[[122, 10, 22, 200], :]
        for a in [None] + list(coxnet.alphas_):
            chf = coxnet.predict_cumulative_hazard_function(xtest, alpha=a)
            assert len(chf) == 4

            sf = coxnet.predict_survival_function(xtest, alpha=a)
            assert len(sf) == 4

    @pytest.mark.parametrize('fn,', ["baseline_survival_", "cum_baseline_hazard_"])
    def test_baseline_models(self, breast_cancer, fn, normalize_options):
        X, y = breast_cancer

        X_test = X.iloc[:19]
        X_train = X.iloc[19:]
        y_train = y[19:]

        coxnet_1 = CoxnetSurvivalAnalysis(
            fit_baseline_model=True, alpha_min_ratio=0.001, n_alphas=23, normalize=normalize_options
        )
        coxnet_1.fit(X_train, y_train)

        coxnet_2 = CoxnetSurvivalAnalysis()
        coxnet_2.set_params(**coxnet_1.get_params())
        Xm = X_train.mean()
        X_train_t = X_train - Xm
        X_test_t = X_test - Xm
        coxnet_2.fit(X_train_t, y_train)

        assert_array_almost_equal(coxnet_1.coef_, coxnet_2.coef_)
        assert_array_almost_equal(coxnet_1.alphas_, coxnet_2.alphas_)

        for k, (a_1, a_2) in enumerate(zip(coxnet_1.alphas_, coxnet_2.alphas_)):
            c0 = coxnet_1.predict(X_test, a_1)
            c1 = coxnet_2.predict(X_test_t, a_2)
            assert_array_almost_equal(c0, c1, err_msg="alphas[{}]".format(k))

            pred_1 = getattr(coxnet_1._baseline_models[k], fn)
            pred_2 = getattr(coxnet_2._baseline_models[k], fn)

            assert_array_almost_equal(pred_1.y, pred_2.y, decimal=3, err_msg="alphas[{}]".format(k))

    @pytest.mark.parametrize('fn', ["predict_survival_function", "predict_cumulative_hazard_function"])
    def test_baseline_predict(self, breast_cancer, fn, normalize_options):
        X, y = breast_cancer

        X_test = X.iloc[:19]
        X_train = X.iloc[19:]
        y_train = y[19:]

        coxnet_1 = CoxnetSurvivalAnalysis(
            fit_baseline_model=True, alpha_min_ratio=0.001, n_alphas=23, normalize=normalize_options
        )
        coxnet_1.fit(X_train, y_train)

        coxnet_2 = CoxnetSurvivalAnalysis()
        coxnet_2.set_params(**coxnet_1.get_params())
        Xm = X_train.mean()
        X_train_t = X_train - Xm
        X_test_t = X_test - Xm
        coxnet_2.fit(X_train_t, y_train)

        assert_array_almost_equal(coxnet_1.coef_, coxnet_2.coef_)
        assert_array_almost_equal(coxnet_1.alphas_, coxnet_2.alphas_)

        time_points = numpy.unique(y["t.tdm"])

        for k, (a_1, a_2) in enumerate(zip(coxnet_1.alphas_, coxnet_2.alphas_)):
            pred_1 = getattr(coxnet_1, fn)(X_test, alpha=a_1)
            pred_2 = getattr(coxnet_2, fn)(X_test_t, alpha=a_2)

            for i, (f1, f2) in enumerate(zip(pred_1, pred_2)):
                assert_array_almost_equal(f1.a, f2.a, 5, err_msg="alphas[{}] [{}].a mismatch".format(k, i))
                assert_array_almost_equal(f1.x, f2.x, err_msg="alphas[{}] [{}].a mismatch".format(k, i))
                assert_array_almost_equal(f1.y, f2.y, err_msg="alphas[{}] [{}].a mismatch".format(k, i))

                out_1 = f1(time_points)
                out_2 = f2(time_points)
                assert_array_almost_equal(out_1, out_2, 5, err_msg="alphas[{}] {}() mismatch".format(k, i))

    def test_predict_func_disabled(self):
        x, coxnet = self._fit_example(l1_ratio=0.9, n_alphas=11,
                                      alpha_min_ratio=0.001,
                                      fit_baseline_model=False)
        with pytest.raises(ValueError,
                           match='`fit` must be called with the fit_baseline_model option set to True.'):
            coxnet.predict_cumulative_hazard_function(x)

    def test_predict_func_no_such_alpha(self):
        x, coxnet = self._fit_example(l1_ratio=0.9, n_alphas=11,
                                      alpha_min_ratio=0.001,
                                      fit_baseline_model=True)
        with pytest.raises(ValueError,
                           match=r'alpha must be one value of alphas_: \[.+'):
            for a in 1. + numpy.random.randn(100):
                coxnet.predict_cumulative_hazard_function(x, alpha=a)

    def test_all_zero_coefs(self):
        alphas = numpy.array([256, 128, 96, 64, 48])

        with pytest.warns(UserWarning, match="all coefficients are zero, consider decreasing alpha."):
            _, coxnet = self._fit_example(l1_ratio=0.9, alphas=alphas,
                                          alpha_min_ratio=0.001)
        assert_array_almost_equal(coxnet.coef_, numpy.zeros((30, 5), dtype=float))

    def test_max_iter(self):
        with pytest.warns(ConvergenceWarning,
                          match=r'Optimization terminated early, you might want'
                                r' to increase the number of iterations \(max_iter=100\).'):
            self._fit_example(alpha_min_ratio=0.0001, l1_ratio=0.9, max_iter=100)

    @pytest.mark.parametrize('val', [0, -1, -1e-6, 1 + 1e-6, 1512, numpy.nan, numpy.infty])
    def test_invalid_l1_ratio(self, val):
        with pytest.raises(ValueError,
                           match=r"l1_ratio must be in interval \]0;1\]"):
            self._fit_example(alpha_min_ratio=0.0001, l1_ratio=val)

    def test_invalid_tol(self, invalid_positive_int):
        with pytest.raises(ValueError,
                           match="tolerance must be positive"):
            self._fit_example(alpha_min_ratio=0.0001, tol=invalid_positive_int)

    def test_invalid_max_iter(self, invalid_positive_int):
        with pytest.raises(ValueError,
                           match="max_iter must be a positive integer"):
            self._fit_example(alpha_min_ratio=0.0001, max_iter=invalid_positive_int)

    def test_invalid_n_alphas(self, invalid_positive_int):
        with pytest.raises(ValueError,
                           match="n_alphas must be a positive integer"):
            self._fit_example(alpha_min_ratio=0.0001, n_alphas=invalid_positive_int)

    @pytest.mark.parametrize('length', [0, 1, 29, 31])
    def test_invalid_penalty_factor_length(self, length):
        msg = r"penalty_factor must be array of length " \
              r"n_features \(30\), but got {:d}".format(length)

        array = numpy.empty(length, dtype=float)
        with pytest.raises(ValueError, match=msg):
            self._fit_example(alpha_min_ratio=0.0001, penalty_factor=array)

    def test_negative_penalty_factor_value(self, negative_float_array):
        with pytest.raises(ValueError,
                           match="Negative values in data passed to penalty_factor"):
            self._fit_example(alpha_min_ratio=0.0001, penalty_factor=negative_float_array)

    def test_invalid_penalty_factor_value(self, infinite_float_array):
        with pytest.raises(ValueError,
                           match="Input contains NaN, infinity or a value too large"):
            self._fit_example(alpha_min_ratio=0.0001, penalty_factor=infinite_float_array)

    def test_negative_alphas(self, negative_float_array):
        with pytest.raises(ValueError,
                           match="Negative values in data passed to alphas"):
            self._fit_example(alpha_min_ratio=0.0001, alphas=negative_float_array)

    def test_invalid_alphas(self, infinite_float_array):
        with pytest.raises(ValueError,
                           match="Input contains NaN, infinity or a value too large"):
            self._fit_example(alpha_min_ratio=0.0001, alphas=infinite_float_array)

    def test_invalid_alpha_min_ratio_string(self):
        with pytest.raises(ValueError,
                           match="Invalid value for alpha_min_ratio"):
            self._fit_example(alpha_min_ratio="max")

    @pytest.mark.parametrize("value", [0.0, -1e-12, -1, -numpy.infty, numpy.nan])
    def test_invalid_alpha_min_ratio_float(self, value):
        with pytest.raises(ValueError,
                           match="alpha_min_ratio must be positive"):
            self._fit_example(alpha_min_ratio=value)

    @staticmethod
    def test_alpha_too_small(breast_cancer):
        Xt, y = breast_cancer

        index = numpy.array([
            0, 1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 33,
            34, 36, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66,
            68, 70, 71, 72, 75, 76, 78, 79, 80, 82, 84, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 98, 99, 100, 102, 103,
            104, 105, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 124, 125, 126, 127, 128,
            130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 143, 144, 145, 147, 148, 150, 151, 153, 154, 155, 156,
            157, 158, 160, 161, 164, 165, 166, 167, 168, 169, 170, 171, 172, 174, 175, 177, 178, 180, 181, 182, 183,
            184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197
        ])

        nn = CoxnetSurvivalAnalysis(alphas=[0.007295025406624247], alpha_min_ratio=0.0001, l1_ratio=1.0)
        Xf, yf = Xt.iloc[index], y[index]

        with pytest.raises(ArithmeticError,
                           match="Numerical error, because weights are too large. Consider increasing alpha."):
            nn.fit(Xf, yf)

    @staticmethod
    def test_breast_example(breast_cancer):
        x, y = breast_cancer

        coxnet = CoxnetSurvivalAnalysis(l1_ratio=1.0)
        coxnet.fit(x.values, y)

        assert coxnet.alpha_min_ratio_ == 0.0001

        expected_alphas = numpy.array([
            0.207764947265866, 0.189307681974955, 0.172490109262135, 0.157166563357949, 0.143204319038428,
            0.130482442022696, 0.118890741498079, 0.108328815700004, 0.0987051822799425, 0.0899364859290742,
            0.0819467763944772, 0.0746668506343715, 0.0680336534144775, 0.0619897311537413, 0.0564827342889011,
            0.051464963847614, 0.046892958302776, 0.0427271171295661, 0.0389313578046448, 0.0354728032765984,
            0.0323214972006479, 0.0294501444711215, 0.0268338748043064, 0.0244500273239498, 0.0222779542835891,
            0.0202988422256499, 0.0184955490282766, 0.0168524554284737, 0.0153553297355215, 0.0139912045628799,
            0.0127482645108893, 0.0116157438274312, 0.0105838331601337, 0.00964359459245389, 0.00878688422772072,
            0.00800628165059773, 0.0072950256549955, 0.0066469556817389, 0.00605645845875073, 0.00551841938157428,
            0.00502817821311635, 0.00458148871890295, 0.00417448188822764, 0.00380363242263169, 0.00346572820145532,
            0.00315784245998521, 0.00287730843921864, 0.00262169628767281, 0.00238879201517371, 0.00217657831633235,
            0.00198321709761059, 0.00180703355663423, 0.00164650167585602, 0.00150023100492174, 0.0013669546172544,
            0.00124551813654232, 0.00113486973808373, 0.00103405103838443, 0.000942188794098442, 0.000858487338411865,
            0.000782221689357606, 0.00071273127036839, 0.000649414188678556, 0.000591722022016858, 0.00053915506843511,
            0.00049125801812897, 0.000447616009762226, 0.000407851037136367, 0.000371618675081733, 0.000338605096211458,
            0.000308524352698783, 0.00028111589953377, 0.000256142337807075, 0.000233387358474159, 0.000212653868789829,
            0.000193762285185162, 0.000176548977800548, 0.000160864853202119, 0.000146574063005757,
            0.000133552827223371, 0.000121688362139862, 0.000110877903434536, 0.000101027816085719,
            9.20527833489927e-05, 8.38750677843702e-05, 7.64238379317803e-05, 6.96345548028444e-05, 6.34484128750348e-05
        ])

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_deviance_ratio = numpy.array([
            0, 0.00741462796207568, 0.0135178719105177, 0.0183232499901932, 0.0221250029051101, 0.0251530137843965,
            0.0275599035016693, 0.0298664819929119, 0.033763232356598, 0.0374249162331977, 0.0409637006907067,
            0.0454486054162627, 0.0551615080395675, 0.0651612844343542, 0.0736024993960834, 0.0808820441173129,
            0.0894426534710234, 0.0992239010000626, 0.108910229105339, 0.121376204780063, 0.134004998770465,
            0.145079557491685, 0.156667501995989, 0.167543840680748, 0.178622131991811, 0.189365153169168,
            0.199027839424271, 0.20909726215036, 0.218610320633419, 0.228024278642459, 0.238171883969976,
            0.248070501745195, 0.258480494697342, 0.268971907277929, 0.280744803445048, 0.291329662029924,
            0.300942928439923, 0.309972153913063, 0.318315812887558, 0.325822700491885, 0.332992506325249,
            0.339665277042211, 0.345876707002969, 0.351605625998246, 0.357206102668659, 0.362484660673399,
            0.367624391654207, 0.372275248793233, 0.37674043994605, 0.380887801196039, 0.384795899779142,
            0.388569806807258, 0.392075171498793, 0.395375481018565, 0.398377579969751, 0.400997300805061,
            0.403375467852471, 0.405431976972633, 0.407443593366561, 0.409668341757423, 0.411628734365416,
            0.413367576771339, 0.414896999887021, 0.416268233594787, 0.417475290203319, 0.418554781508749,
            0.419526121036389, 0.420522904669104, 0.421455233639571, 0.422296101083462, 0.423049677446171,
            0.423716974236606, 0.424302533927477, 0.424825925226932, 0.425286695396174, 0.425693415010937,
            0.426052733081791, 0.426369464812111, 0.426652822940747, 0.42686317150694, 0.427072533094355,
            0.427264216646862, 0.427427314063872, 0.427573225735422, 0.427700379783919, 0.427814235325525,
            0.427912925916531, 0.427998148400703
        ])

        assert_array_almost_equal(expected_deviance_ratio, coxnet.deviance_ratio_)

        coef = pandas.DataFrame(coxnet.coef_, index=x.columns, dtype=float)
        expected_coef = pandas.read_csv(BREAST_CANCER_COEFFICIENTS_FILE, index_col=0)
        expected_coef.columns = numpy.arange(expected_coef.shape[1])

        assert_columns_almost_equal(coef, expected_coef, 5)

    @staticmethod
    def test_breast_p_greater_n(breast_cancer):
        x, y = breast_cancer
        x -= x.mean()
        order = (-y['t.tdm']).argsort()
        x = x.iloc[order[:80]]
        y = y[order[:80]]

        coxnet = CoxnetSurvivalAnalysis(l1_ratio=1.0)
        coxnet.fit(x.values, y)

        assert coxnet.alpha_min_ratio_ == 0.01

        expected_alphas = numpy.array([
            0.0948288922619048, 0.0905187727555525, 0.086404554832736, 0.0824773344641409, 0.0787286123223276,
            0.0751502753874057, 0.0717345793887599, 0.0684741320448266, 0.0653618770646479, 0.0623910788765801,
            0.0595553080511063, 0.0568484273862036, 0.054264578625153, 0.0517981697780446, 0.0494438630195401,
            0.0471965631367011, 0.0450514065018809, 0.0430037505468153, 0.0410491637151331, 0.0391834158715392,
            0.0374024691469166, 0.0357024691995311, 0.0340797368734291, 0.0325307602359731, 0.0310521869772843,
            0.0296408171551414, 0.0282935962696359, 0.0270076086525939, 0.0257800711574603, 0.024608327135986,
            0.023489840688685, 0.0224221911766166, 0.021403067982616, 0.0204302655106344, 0.0195016784123676,
            0.0186152970308409, 0.0177692030510908, 0.0169615653485295, 0.0161906360260076, 0.0154547466309991,
            0.0147523045447198, 0.0140817895353677, 0.013441750468022, 0.0128308021640839, 0.0122476224034596,
            0.0116909490629993, 0.0111595773849981, 0.0106523573698482, 0.0101681912871989, 0.00970603130023803,
            0.00926487719795367, 0.00884377423046809, 0.00844181104275901, 0.00805811770229645, 0.00769186381632648,
            0.00734225673472744, 0.00700853983454904, 0.00668999088252177, 0.0063859204719929, 0.00609567053090603,
            0.00581861289759537, 0.00555414796131231, 0.00530170336454213, 0.00506073276430255, 0.00483071464974297,
            0.00461115121348587, 0.00440156727426728, 0.00420150924854507, 0.00401054416884912, 0.00382825874674904,
            0.00365425847841146, 0.00348816679081109, 0.00332962422674778, 0.00317828766690595, 0.00303382958727242,
            0.00289593735030589, 0.00276431252832385, 0.00263867025764251, 0.00251873862207214, 0.00240425806443351,
            0.00229498082482178, 0.00219067040440213, 0.00209110105357681, 0.00199605728341568, 0.00190533339929305,
            0.00181873305572143, 0.00173606883141875, 0.00165716182368948, 0.00158184126124171, 0.00150994413460228,
            0.00144131484333019, 0.00137580485926463, 0.0013132724050789, 0.00125358214744464, 0.00119660490414211,
            0.00114221736448283, 0.00109030182243944, 0.00104074592190515, 0.000993442413531659, 0.000948288922619052
        ])

        assert_array_almost_equal(expected_alphas, coxnet.alphas_)

        expected_deviance_ratio = numpy.array([
            0, 0.0462280671677718, 0.0886824292023055, 0.127809395879191, 0.167202029735547, 0.21150983511365,
            0.25202048418497, 0.288977870766515, 0.322894526112143, 0.354184937130736, 0.383227342294078,
            0.41020873570105, 0.435399917854983, 0.458992629893996, 0.481148794953564, 0.502007028421266,
            0.521687007555194, 0.540326070242061, 0.557951417089716, 0.574673796628669, 0.590565196035048,
            0.605720094024342, 0.620136650121787, 0.633893317242588, 0.647036975167954, 0.659641706497287,
            0.671685014319545, 0.683229207693947, 0.694339713809272, 0.704984204297195, 0.715217033589695,
            0.725098421441951, 0.734586489938626, 0.743749179587877, 0.756951344840625, 0.769374262091115,
            0.781088604662055, 0.792115179997869, 0.802515292743199, 0.812333737139071, 0.821610165074286,
            0.830380590378993, 0.838677922696392, 0.846532328413876, 0.853971528730985, 0.861021056012275,
            0.867704476556531, 0.874043585172118, 0.880058575813189, 0.885768191772895, 0.891189858340372,
            0.89633980035321, 0.901233146689909, 0.905884008467536, 0.910305596711285, 0.914510324284785,
            0.918509724810968, 0.922314672256325, 0.925935372520073, 0.929371697720327, 0.932660004492162,
            0.935775430981201, 0.938748309596168, 0.941580730549356, 0.944278997138997, 0.946849613332775,
            0.949298889398519, 0.951632778072432, 0.953856965822007, 0.955976832090688, 0.957995949491087,
            0.95992200533877, 0.961758223454499, 0.963507897227564, 0.96517707093905, 0.966769081504476,
            0.968287420437941, 0.969735526668442, 0.971107538859051, 0.972432535259795, 0.973681544039006,
            0.97487938238315, 0.976022813266596, 0.977113969626666, 0.978155133509104, 0.979151911611324,
            0.980095030270948, 0.981003596012864, 0.98186553640233, 0.982688707756662, 0.983474488924476,
            0.984224494959766, 0.984940315524502, 0.985612152584937, 0.98627427317818, 0.986896672934578,
            0.987488984924729, 0.988056860176794, 0.988587033996462, 0.989113296247043
        ])

        assert_array_almost_equal(expected_deviance_ratio, coxnet.deviance_ratio_)

        coef = pandas.DataFrame(coxnet.coef_, index=x.columns, dtype=float)
        expected_coef = pandas.read_csv(BREAST_CANCER_HIGH_COEFFICIENTS_FILE, index_col=0)
        expected_coef.columns = numpy.arange(expected_coef.shape[1])

        assert_columns_almost_equal(coef, expected_coef, 5)

    @staticmethod
    def test_simple():
        y = Surv.from_arrays([True, False, False, True, False], [7., 8., 11., 11., 23.],
                             name_event="D", name_time="Y")

        x = pandas.DataFrame({"F1": [1, 1, 1, 0, 0],
                              "F2": [23, 43, 54, 75, 67],
                              "F3": [120, 98, 78, 91, 79],
                              "F4": [0.123, 0.541, 0.784, 0.846, 0.331]})

        coxnet = CoxnetSurvivalAnalysis(alpha_min_ratio=0.0001, l1_ratio=1.0)
        coxnet.fit(x.values, y)

        expected_alphas = numpy.array(
            [7.02666666666667, 6.40243696630484, 5.83366211207401, 5.31541564828386, 4.84320877198972, 4.41295145312887,
             4.02091700863675, 3.66370982370111, 3.3382359405709, 3.04167626017436, 2.77146212443153, 2.52525306776672,
             2.30091654511542, 2.09650946083909, 1.91026133856035, 1.74055898614351, 1.5859325229961, 1.44504264866632,
             1.31666904246323, 1.19969979362274, 1.09312177046848, 0.996011845149902, 0.907528897950459,
             0.826906531910992, 0.753446434665921, 0.686512329995589, 0.625524466706047, 0.569954597101554,
             0.519321401555745, 0.473186319551291, 0.431149751078499, 0.392847595491192, 0.357948097841098,
             0.326148975375191, 0.297174799307102, 0.270774609184727, 0.24671973919085, 0.22480183754923,
             0.204831061881182, 0.186634434881721, 0.170054346072885, 0.154947186657187, 0.141182105646904,
             0.128639876495421, 0.117211864413924, 0.106799085428826, 0.0973113490299429, 0.0886664769834391,
             0.0807895915432809, 0.0736124668960205, 0.0670729382214382])

        # FIXME
        assert_array_almost_equal(expected_alphas, coxnet.alphas_[:len(expected_alphas)])

        coef = pandas.DataFrame(coxnet.coef_[:, :len(expected_alphas)],
                                dtype=float)
        expected_coef = pandas.read_csv(SIMPLE_COEF_FILE, header=None, skiprows=1)

        assert_columns_almost_equal(coef, expected_coef)


@pytest.mark.parametrize("func", ("predict_survival_function", "predict_cumulative_hazard_function"))
def test_pipeline_predict(breast_cancer, func):
    X_str, _ = load_breast_cancer()
    X_num, y = breast_cancer

    est = CoxnetSurvivalAnalysis(alpha_min_ratio=0.0001, l1_ratio=1.0, fit_baseline_model=True)
    est.fit(X_num[10:], y[10:])

    pipe = make_pipeline(OneHotEncoder(), CoxnetSurvivalAnalysis(alpha_min_ratio=0.0001,
                                                                 l1_ratio=1.0,
                                                                 fit_baseline_model=True))
    pipe.fit(X_str[10:], y[10:])

    tree_pred = getattr(est, func)(X_num[:10])
    pipe_pred = getattr(pipe, func)(X_str[:10])

    for s1, s2 in zip(tree_pred, pipe_pred):
        assert_array_almost_equal(s1.x, s2.x)
        assert_array_almost_equal(s1.y, s2.y)
