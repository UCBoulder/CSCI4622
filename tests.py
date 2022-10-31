import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from warnings import simplefilter
simplefilter("ignore", category=ConvergenceWarning)

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 100


class Tester(object):
    def __init__(self):
        self.questions = {}

    def add_test(self, question, test_function):
        self.questions[question] = test_function

    def run(self):
        for question in self.questions:
            success, comment = self.questions[question]()
            if success:
                print("Question %s: [PASS]" % question)
            else:
                print("Question %s: [FAIL]" % question, comment)


def test_ridge_coef(Ridge, normalize):
    tester = Tester()
    features = np.array([[1.55143777, 0.2644804, 0.0995576],
                         [0.22541014, 1.6967911, -0.45701382],
                         [0.12528546, -1.44263567, 0.7017054],
                         [-1.30567135, -0.86010032, -1.13522536]])
    labels = np.array([136.70039877, 10.1003086, 44.67363091, -221.48398972])

    coef_ = np.array([67.3816571, 12.4267024, 46.63028522])
    if normalize:
        coef_ = np.array([59.85861897, 18.28561265, 48.08714515])
    if normalize is None:
        reg = Ridge(alpha=2)
    else:
        reg = Ridge(alpha=2, normalize=normalize)
    reg.fit(features, labels)

    ins = "\n X:" + str(features) + ",\ntargets:" + str(labels)
    topic = "Testing Ridge's fit,  normalize: {}, alpha: 2.0".format(normalize)

    def test_coef():
        outs = coef_
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = reg.coefficients
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.1.a", test_coef)
    tester.run()


def test_ridge_intercept(Ridge, normalize):
    tester = Tester()
    features = np.array([[1.55143777, 0.2644804, 0.0995576],
                         [0.22541014, 1.6967911, -0.45701382],
                         [0.12528546, -1.44263567, 0.7017054],
                         [-1.30567135, -0.86010032, -1.13522536]])
    labels = np.array([136.70039877, 10.1003086, 44.67363091, -221.48398972])

    intercept = -7.2683820675025785
    if normalize:
        intercept = -7.502412860000002
    if normalize is None:
        reg = Ridge(alpha=2)
    else:
        reg = Ridge(alpha=2, normalize=normalize)

    reg.fit(features, labels)

    ins = "\n X:" + str(features) + ",\ntargets:" + str(labels)
    topic = "Testing Ridge's intercept,  normalize: {}, alpha: 2.0".format(normalize)

    def test_coef():
        outs = intercept
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = reg.intercept
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.1.b", test_coef)
    tester.run()


def test_lasso_coef(Lasso, normalize):
    tester = Tester()
    features = np.array([[1.55143777, 0.2644804, 0.0995576],
                         [0.22541014, 1.6967911, -0.45701382],
                         [0.12528546, -1.44263567, 0.7017054],
                         [-1.30567135, -0.86010032, -1.13522536]])
    labels = np.array([136.70039877, 10.1003086, 44.67363091, -221.48398972])

    coef_ = np.array([83.36110924, 15.21050409, 79.08888918])
    if normalize:
        coef_ = np.array([82.32525113, 19.25895655, 56.15183344])

    if normalize is None:
        reg = Lasso(alpha=2)
    else:
        reg = Lasso(alpha=2, normalize=normalize)
    reg.fit(features, labels)

    ins = "\n X:" + str(features) + ",\ntargets:" + str(labels)
    topic = "Testing Lasso's fit,  normalize: {}, alpha: 2.0".format(normalize)

    def test_coef():
        outs = coef_
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = reg.coefficients
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.3.a", test_coef)
    tester.run()


def test_lasso_intercept(Lasso, normalize):
    tester = Tester()
    features = np.array([[1.55143777, 0.2644804, 0.0995576],
                         [0.22541014, 1.6967911, -0.45701382],
                         [0.12528546, -1.44263567, 0.7017054],
                         [-1.30567135, -0.86010032, -1.13522536]])
    labels = np.array([136.70039877, 10.1003086, 44.67363091, -221.48398972])

    intercept = -2.9950281444221063
    if normalize:
        intercept = -7.502412860000005

    if normalize is None:
        reg = Lasso(alpha=2)
    else:
        reg = Lasso(alpha=2, normalize=normalize)
    reg.fit(features, labels)

    ins = "\n X:" + str(features) + ",\ntargets:" + str(labels)
    topic = "Testing Lasso's intercept, normalize: {}, alpha: 2.0".format(normalize)

    def test_coef():
        outs = intercept
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = reg.intercept
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.3.b", test_coef)
    tester.run()

def test_rmse(rmse):
    tester = Tester()
    X1 = np.array([0.59872203, 1.84268893, 0.86998721, -0.42818316, -0.83615123])
    X2 = np.array([0.34545115, 0.42796338, -0.6454915, -0.72670301, 0.4195794])
    ins = "\n X1:" + str(X1) + ",\nX2:" + str(X2)
    topic = "Testing rmse(X1,X2)"

    def test_r():
        outs = 1.0980203783998055
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = rmse(X1, X2)
        comment = comment + "\n obtained: " + str(obtained)
        if np.allclose(obtained, outs, atol=1e-5):
            return True, comment
        return False, comment

    tester.add_test("1.1.c", test_r)
    tester.run()
