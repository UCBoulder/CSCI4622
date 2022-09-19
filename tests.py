import unittest
import sys
import numpy as np
import os

current_folder = os.path.dirname(os.path.abspath(__file__))


def _compare_node(node1, node2):
    # Not equal if one is terminal and the other is not
    if node1.is_terminal() != node2.is_terminal():
        return False
    # leaf nodes are equal if they have the same label
    if node1.is_terminal():
        return node1.label == node2.label
    # parent nodes are equal if they have the same feature_id, threshold, and equal children
    compare_values = (node1.feature_id == node2.feature_id) and np.isclose(node1.threshold, node2.threshold, atol=1e-5)
    if not compare_values:
        return False
    return _compare_node(node1.left, node2.left) and _compare_node(node1.right, node2.right)


def compare_trees(tree1, tree2):
    return _compare_node(tree1.tree, tree2.tree)

class TestDT(unittest.TestCase):

    def test_leaf(self):
        # print("majority vote test")
        labels = np.array([1, -1, -1, 1, -1, 1, -1, 2, 2, 2])
        ins = "\nlabels:" + str(labels)
        topic = "Testing LeafNode "
        outs = -1
        comment = topic + "compute_label" + ins + "\n expected output:  \n" + str(outs)
        leaf = LeafNode(labels)
        obtained = leaf.label
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, outs, atol=1e-5), msg=comment)

    def test_entropy(self):
        labels = np.array([1, 1, 2, 2, 3, 3, 3, 3])
        ins = "\nlabels:" + str(labels)
        topic = "Testing Entropy Function "

        outs = 1.5
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = entropy(labels)
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, outs, atol=1e-5), msg=comment)

    def test_reduction(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        left_indices = np.array([0, 1, 3])
        right_indices = np.array([2, 4, 5])

        ins = "\nlabels:" + str(labels) + "\nleft_indices:" + str(left_indices) + "\nright_indices" + str(right_indices)
        topic = "Testing Information Gain Function "

        outs = 4.0
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = uncertainty_reduction(labels, left_indices, right_indices)
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, outs, atol=1e-5), msg=comment)

    def test_split(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        features = np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 0]
        ])

        ins = "\n X:" + str(features) + ",\nlabels:" + str(labels)
        topic = "Testing Best Partition Function "

        outs = (2, 0.5)
        comment = topic + ins + "\n expected output:  \n" + str(outs)
        obtained = best_split(features, labels)[:2]
        comment = comment + "\n obtained: " + str(obtained)
        self.assertTrue(np.allclose(obtained, outs, atol=1e-5), msg=comment)


    def test_discretize(self):

        topic = "Testing house prices categories based on mean/std"
        mean =1.6097393689986284
        std =  1.1169420153612106
        comment = f"{topic}\n mean:{mean}, std: {std}"
        obtained = (np.mean(house_prices.y), np.std(house_prices.y))
        comment = f"{comment}\n obtained (mean, std) {obtained}"
        self.assertTrue(np.allclose(obtained, (mean, std), atol=1e-5), msg=comment)

    def test_importance(self):
        topic = "feature importance test"
        tree = DecisionTree(max_depth=6, min_samples_split=3).fit(features, labels)
        importance = tree.feature_importance(features, labels)
        expected = [0.05624964746721033, 0.5971439443261835, 0.3466064082066061, 0.0]
        obtained = [importance[k] for k in range(4)]
        comment = f"{topic}\n expected:{expected}\n obtained:{obtained}"
        self.assertTrue(np.allclose(expected, obtained, atol=1e-5), msg=comment)



    def test_build(self):
        import helpers
        features_names = ["age", "income", "single", "has_pets"]
        correct_tree = np.load(os.path.join(current_folder, "data/tree_depth3_min2.npy"), allow_pickle=True)[0]
        topic = "Testing DecisionTree build with depth 3 and min_samples_split 2"
        expected_tree = "\n".join(helpers.node_to_string(correct_tree.tree, features_names)[0])
        ins = ", using Problem1 features, and labels"
        tree = DecisionTree(max_depth=3, min_samples_split=2,).fit(features, labels)
        obtained_tree = "\n".join(helpers.node_to_string(tree.tree, features_names)[0])
        comment = topic + ins + "\n expected output: \n" + expected_tree + "\n obtained:\n " + obtained_tree
        self.assertTrue(compare_trees(correct_tree, tree), msg=comment)


    def test_projection(self):
        Q = random_projection(10,4)
        comment = "Testing that Q's columns are orthonormal: Q.T@Q = I for (10, 4)"
        expected = np.eye(4)
        obtained = Q.T@Q
        comment  = f"{comment}\n expected:\n {expected},\n obtained:\n{obtained}"
        self.assertTrue(np.allclose(obtained, expected, atol=1e-6), msg=comment)



test = sys.argv[1]

dt_tests = ["test_leaf", "test_predict", "test_confusion", "test_accuracy"]
perceptron_tests = ["test_weights", "test_mistakes"]
suite = unittest.TestSuite()

# if test == "knn":
#     for t in KNN_tests:
#         suite.addTest(TestKNNClassifier(t))
# elif test == "perceptron":
#     for t in perceptron_tests:
#         suite.addTest(TestPerceptron(t))

suite.addTest(TestDT(f"test_{test}"))
runner = unittest.TextTestRunner(verbosity=1).run(suite)
