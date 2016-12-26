from numpy.testing import TestCase, run_module_suite

from sksurv.bintrees import RBTree, AVLTree


class BinaryTreesCases(object):
    TREE_CLASS = None

    def test_insert(self):
        tree = self.TREE_CLASS(10)
        for k in [12, 34, 45, 16, 35, 57]:
            tree.insert(k, k)
        self.assertEqual(6, len(tree))

    def test_count_smaller(self):
        tree = self.TREE_CLASS(10)
        for k in [12, 34, 45, 16, 35, 57]:
            tree.insert(k, k)

        c, a = tree.count_smaller(12)
        self.assertEqual(0, c)

        c, a = tree.count_smaller(16)
        self.assertEqual(1, c)

        c, a = tree.count_smaller(34)
        self.assertEqual(2, c)

        c, a = tree.count_smaller(35)
        self.assertEqual(3, c)

        c, a = tree.count_smaller(45)
        self.assertEqual(4, c)

        c, a = tree.count_smaller(57)
        self.assertEqual(5, c)

    def test_count_larger(self):
        tree = self.TREE_CLASS(10)
        for k in [12, 34, 45, 16, 35, 57]:
            tree.insert(k, k)

        c, a = tree.count_larger(12)
        self.assertEqual(5, c)

        c, a = tree.count_larger(16)
        self.assertEqual(4, c)

        c, a = tree.count_larger(34)
        self.assertEqual(3, c)

        c, a = tree.count_larger(35)
        self.assertEqual(2, c)

        c, a = tree.count_larger(45)
        self.assertEqual(1, c)

        c, a = tree.count_larger(57)
        self.assertEqual(0, c)


class TestRBTree(BinaryTreesCases, TestCase):
    TREE_CLASS = RBTree


class TestAVLTree(BinaryTreesCases, TestCase):
    TREE_CLASS = AVLTree


if __name__ == '__main__':
    run_module_suite()
