import numpy as np
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

# Unicodes used to print the tree
VERTICAL = "\u2502"
UP_IN = "\u250C"
DOWN_IN = "\u2514"
OUT_UP = "\u2518"
OUT_DOWN = "\u2510"


def node_to_string(root_node, features_names):
    if root_node is None:
        return "None"
    if root_node.is_terminal():
        return [VERTICAL + "label: %i" % root_node.label], 0
    else:
        string = ["%s" % features_names[root_node.feature_id], "%.2f" % root_node.threshold]
        max_len = max([len(s) for s in string])
        string[0] = "|" + string[0] + " " * (max_len - len(string[0])) + VERTICAL + "L"  # OUT_UP
        string[1] = "|" + string[1] + " " * (max_len - len(string[1])) + VERTICAL + "R"  # OUT_DOWN
        left, left_pos = node_to_string(root_node.left, features_names)
        right, right_pos = node_to_string(root_node.right, features_names)

        for i in range(0, left_pos):
            left[i] = " " + left[i]
        left[left_pos] = UP_IN + left[left_pos]
        for i in range(left_pos + 1, len(left)):
            left[i] = VERTICAL + left[i]

        for i in range(0, right_pos):
            right[i] = VERTICAL + right[i]
        right[right_pos] = DOWN_IN + right[right_pos]
        for i in range(right_pos + 1, len(right)):
            right[i] = " " + right[i]
        left = [" " * (max_len + 2) + l_str for l_str in left]
        right = [" " * (max_len + 2) + r for r in right]
        return left + string + right, len(left)



def print_tree(decision_tree, features_names=None):
    if features_names is None:
        features_names = ["feat_%i" % i for i in range(decision_tree.num_features)]
    print("\n".join(node_to_string(decision_tree.tree, features_names)[0]))
