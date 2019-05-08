/**
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef GLMNET_IRANGE_H
#define GLMNET_IRANGE_H

namespace coxnet {

template<typename T>
class range_iterator {
public:
    typedef range_iterator<T> iterator;
    typedef T value_type;

    const iterator& operator++() {
        ++m_value;
        return *this;
    }

    const value_type& operator*() const {
        return m_value;
    }

    friend
    bool operator==(const iterator& __x, const iterator& __y) {
        return __x.m_value == __y.m_value;
    }
    friend
    bool operator!=(const iterator& __x, const iterator& __y) {
        return __x.m_value != __y.m_value;
    }

    explicit range_iterator(const T value) : m_value(value) {}

private:
    T m_value;
};


template<typename T>
class irange {
public:
    typedef range_iterator<T> iterator;

    irange(const T first, const T last) : m_begin(first), m_end(last)
    {
        eigen_assert(first <= last);
    }

    const iterator& begin() const {
        return m_begin;
    }

    const iterator& end() const {
        return m_end;
    }

private:
    iterator m_begin;
    iterator m_end;
};

}

#endif //GLMNET_IRANGE_H