import unittest
import sys
import numpy as np


class TestKNNClassifier(unittest.TestCase):

    def setUp(self):
        self.features = np.array([[1, 1], [1, 2], [2, 1], [5, 2], [3, 2], [8, 2], [2, 4]])
        self.labels = np.array([1, -1, -1, 1, -1, 1, -1])
        self.test_points = np.array([[1, 1.1], [3, 1], [7, 5], [2, 6], [4, 4]])
        self.ins = "\n X:" + str(self.features) + ",\nlabels:" + str(self.labels)
        self.topic = "Testing KNN(3) "
        self.knn_3 = KNNClassifier(3).fit(self.features, self.labels)

    def test_majority(self):
        # print("majority vote test")
        majority = np.array([-1, 1])
        comment = self.topic + "majority_vote" + self.ins + "\n expected output: " + str(majority)
        obtained = self.knn_3.majority_vote(np.array([[1, 2, 3], [3, 4, 5]]), np.array([[.1, .2, .3], [.1, .2, .3]]))
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(majority, obtained), msg=comment)

    def test_predict(self):
        test_labels = np.array([-1, -1, 1, -1, -1])
        comment = f"{self.topic}  predict {self.ins} \n expected output: {test_labels}"
        obtained = self.knn_3.predict(self.test_points)
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(test_labels, obtained), msg=comment)

    def test_confusion(self):
        confusion = np.array([[2., 1.], [2., 0.]])
        comment = f"{self.topic} confusion {self.ins} \n expected output:  {confusion}"
        obtained = self.knn_3.confusion_matrix(self.test_points, np.array([1, -1, -1, 1, -1]))
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(confusion, obtained), msg=comment)

    def test_accuracy(self):
        # print("Accuracy test")
        accuracy = 0.6
        comment = self.topic + "accuracy" + self.ins + "\n expected output: " + str(accuracy)
        obtained = self.knn_3.accuracy(self.test_points, np.array([1, 1, 1, -1, -1]))
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, accuracy), msg=comment)


class TestPerceptron(unittest.TestCase):

    def setUp(self) -> None:
        self.perceptron = Perceptron()
        np.random.seed(42)
        self.topic = "Testing Perceptron"

    def test_weights(self):
        self.data = data.SeparableData(num_samples=5, margin=0.1, random_seed=43)
        self.X_train, self.y_train = self.data.X_train, self.data.y_train
        self.ins = "\n X:" + str(self.X_train) + ",\nlabels:" + str(self.y_train)
        self.perceptron.fit(self.X_train, self.y_train)
        obtained = self.perceptron.w
        expected = np.array([0.02693284, 1.29053838, 1.])
        comment = self.topic + "weights" + self.ins + "\n expected output: " + str(expected)
        comment = f"{comment} \n obtained {obtained}"
        self.assertTrue(np.allclose(expected, obtained, atol=1e-5), msg=comment)

    def test_mistakes(self):
        self.data = data.SeparableData(num_samples=7, margin=0.1, random_seed=43)
        self.X_train, self.y_train = self.data.X_train, self.data.y_train
        self.ins = "\n X:" + str(self.X_train) + ",\nlabels:" + str(self.y_train)
        self.perceptron.fit(self.X_train, self.y_train)
        obtained = self.perceptron.num_mistakes
        expected = 3
        comment = self.topic + "weights" + self.ins + "\n expected output: " + str(expected)
        comment = f"{comment} \n obtained {obtained}"
        self.assertTrue(np.allclose(expected, obtained, atol=1e-5), msg=comment)



test = sys.argv[1]

KNN_tests = ["test_majority", "test_predict", "test_confusion", "test_accuracy"]
perceptron_tests = ["test_weights", "test_mistakes"]
suite = unittest.TestSuite()

if test == "knn":
    for t in KNN_tests:
        suite.addTest(TestKNNClassifier(t))
elif test == "perceptron":
    for t in perceptron_tests:
        suite.addTest(TestPerceptron(t))
runner = unittest.TextTestRunner(verbosity=1).run(suite)
