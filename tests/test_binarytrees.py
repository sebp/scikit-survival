import pytest

from sksurv.bintrees import RBTree, AVLTree


@pytest.fixture(params=[RBTree, AVLTree])
def tree(request):
    return request.param(10)


class TestBinaryTree(object):
    @staticmethod
    def test_insert(tree):
        for k in [12, 34, 45, 16, 35, 57]:
            tree.insert(k, k)
        assert 6 == len(tree)

    @staticmethod
    def test_count_smaller(tree):
        for k in [12, 34, 45, 16, 35, 57]:
            tree.insert(k, k)

        c, a = tree.count_smaller(12)
        assert 0 == c

        c, a = tree.count_smaller(16)
        assert 1 == c

        c, a = tree.count_smaller(34)
        assert 2 == c

        c, a = tree.count_smaller(35)
        assert 3 == c

        c, a = tree.count_smaller(45)
        assert 4 == c

        c, a = tree.count_smaller(57)
        assert 5 == c

    @staticmethod
    def test_count_larger(tree):
        for k in [12, 34, 45, 16, 35, 57]:
            tree.insert(k, k)

        c, a = tree.count_larger(12)
        assert 5 == c

        c, a = tree.count_larger(16)
        assert 4 == c

        c, a = tree.count_larger(34)
        assert 3 == c

        c, a = tree.count_larger(35)
        assert 2 == c

        c, a = tree.count_larger(45)
        assert 1 == c

        c, a = tree.count_larger(57)
        assert 0 == c
