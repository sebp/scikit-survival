// http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#large_scale_ranksvm
//
// Copyright (c) 2013 Chih-Jen Lin and Ching-Pei Lee
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither name of copyright holders nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#ifndef _BINARYTREES
#define _BINARYTREES

enum {RED,BLACK};
enum {LEFT,RIGHT};
struct node
{
    node* parent;
    node* child[2];
    double key;
    int size;
    bool color;
    int height;
    double vector_sum;
};

class rbtree
{
public:
    rbtree(int l);
    virtual ~rbtree();
    virtual void insert_node(double key, double value);
    void count_larger(double key, int* count_ret, double* acc_value_ret) const;
    void count_smaller(double key, int* count_ret, double* acc_value_ret) const;
    double vector_sum_larger(double key) const;
    double vector_sum_smaller(double key) const;
    int get_size() const { return tree_size;}
protected:
    node* null_node;
    int tree_size;
    void rotate(node* x, int direction);
    virtual void tree_color_fix(node* x);
    node* root;
    node* tree_nodes;
};


class avl: public rbtree
{
public:
    avl(int l);
    virtual void insert_node(double key, double value);
private:
    void tree_balance_fix(node* x);
};

class aatree: public rbtree
{
public:
    aatree(int l):rbtree(l){};
protected:
    virtual void tree_color_fix(node* x);
};

#endif