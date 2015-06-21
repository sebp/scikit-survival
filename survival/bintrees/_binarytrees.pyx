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
from libcpp cimport bool
from libcpp.cast cimport dynamic_cast


cdef extern from "binarytrees.h":
    cdef cppclass rbtree:
        rbtree(int l)
        void insert_node(double key, double value)
        void count_larger(double key, int *count_ret, double *acc_value_ret)
        void count_smaller(double key, int *count_ret, double *acc_value_ret)
        double vector_sum_larger(double key)
        double vector_sum_smaller(double key)
        int get_size()

    cdef cppclass avl(rbtree):
        avl(int l)

    cdef cppclass aatree(rbtree):
        aatree(int l)


ctypedef rbtree* rbtree_ptr


cdef class BaseTree:
    cdef rbtree_ptr treeptr

    def __dealloc__(self):
        if self.treeptr is not NULL:
            del self.treeptr
            self.treeptr = NULL

    def __len__(self):
        return self.treeptr.get_size()

    def insert(self, double key, double value):
        self.treeptr.insert_node(key, value)

    def count_smaller(self, double key):
        cdef int count_ret;
        cdef double acc_value_ret;

        self.treeptr.count_smaller(key, &count_ret, &acc_value_ret)

        return count_ret, acc_value_ret

    def count_larger(self, double key):
        cdef int count_ret
        cdef double acc_value_ret

        self.treeptr.count_larger(key, &count_ret, &acc_value_ret)

        return count_ret, acc_value_ret

    def vector_sum_smaller(self, double key):
        return self.treeptr.vector_sum_smaller(key)

    def vector_sum_larger(self, double key):
        return self.treeptr.vector_sum_larger(key)

    def count_larger_with_event(self, double key, bool has_event):
        if not has_event:
            return 0, 0.0
        return self.count_larger(key)


cdef class RBTree(BaseTree):
    def __cinit__(self, int size):
        if size <= 0:
            raise ValueError('size must be greater zero')
        self.treeptr = new rbtree(size)

cdef class AVLTree(BaseTree):
    def __cinit__(self, int size):
        if size <= 0:
            raise ValueError('size must be greater zero')
        self.treeptr = dynamic_cast[rbtree_ptr](new avl(size))

cdef class AATree(BaseTree):
    def __cinit__(self, int size):
        if size <= 0:
            raise ValueError('size must be greater zero')
        self.treeptr = dynamic_cast[rbtree_ptr](new aatree(size))
