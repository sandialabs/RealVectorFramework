#pragma once

#include <vector>
#include "advanced/binary_in_place.hpp"

namespace rvf {

template<class T, class F>
void tag_invoke(binary_in_place_ftor, 
                std::vector<T>& y, 
                F&& f,
                const std::vector<T>& x) {
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = f(y[i], x[i]);
    }
}

} // namespace rvf
