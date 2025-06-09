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
#ifndef GLMNET_ORDERED_DICT_H
#define GLMNET_ORDERED_DICT_H

#include <map>
#include <memory>
#include <set>


namespace coxnet {

template <typename Key>
struct __link {
    using key_type = Key;
    using link_type = __link<key_type>;
    using pointer = std::shared_ptr<link_type>;

    key_type key;
    pointer next;
    std::weak_ptr<link_type> prev;

    explicit __link() = default;
    __link (const Key &_key) : key{_key} {}
    __link (const Key &_key,
            pointer &_next,
            pointer &_prev) : key{_key}, next{_next}, prev{_prev} {}
};

template <typename T>
class ordered_dict_iterator {
public:
    using link_type = T;
    using link_pointer = std::shared_ptr<link_type>;
    using key_type = typename link_type::key_type;
    using iterator = ordered_dict_iterator<T>;

    ordered_dict_iterator(const link_pointer &__root) : m_root{__root} {}

    iterator& operator++() {
        link_pointer curr = m_root->next;
        m_root = curr;
        return *this;
    }

    const key_type& operator*() const {
        return m_root->key;
    }

    friend
    bool operator==(const iterator& __x, const iterator& __y) {
        return __x.m_root == __y.m_root;
    }
    friend
    bool operator!=(const iterator& __x, const iterator& __y) {
        return !(__x.m_root == __y.m_root);
    }

private:
    link_pointer m_root;
};

template <typename Key>
class ordered_dict : public std::set<Key> {
public:
    using base = std::set<Key>;
    using key_type = typename base::key_type;
    using link_type = __link<Key>;
    using link_pointer = std::shared_ptr<link_type>;
    using const_iterator = ordered_dict_iterator<const link_type>;

    explicit ordered_dict() : m_root { std::make_shared<link_type>(-1) } {
        m_root->next = m_root;
        m_root->prev = m_root;
    }

    void insert_ordered( const key_type &key );

    const_iterator cbegin_ordered() const {
        return const_iterator(m_root->next);
    }
    const_iterator cend_ordered() const {
        return const_iterator(m_root);
    }

private:
    std::map<key_type, link_pointer> m_map;
    link_pointer m_root;
};


template <typename Key>
void ordered_dict<Key>::insert_ordered( const key_type &key )
{
    auto search = this->find(key);
    if (search == this->end()) {
        link_pointer last { m_root->prev.lock() };
        link_pointer new_link = std::make_shared<link_type>(key, m_root, last);
        last->next = new_link;
        m_root->prev = new_link;

        m_map.emplace(std::make_pair(key, new_link));
    }
    this->insert(key);
};

};

#endif //GLMNET_ORDERED_DICT_H
