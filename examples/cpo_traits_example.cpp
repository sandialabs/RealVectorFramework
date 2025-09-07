/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#include <iostream>
#include <vector>
#include <string>
#include <type_traits>

// Example: A serialize CPO that needs different strategies for different types

namespace framework {

template<typename T>
struct cpo_impl; // Primary template - no definition means "not supported"

struct serialize_ftor {
    template<typename T>
    requires requires { cpo_impl<T>::call(std::declval<T>()); }
    auto operator()(const T& value) const -> decltype(cpo_impl<T>::call(value)) {
        return cpo_impl<T>::call(value);
    }
};
inline constexpr serialize_ftor serialize{};

// Traits approach allows us to:
// 1. Query support at compile-time
// 2. Centralize complex type-specific logic
// 3. Provide defaults and specializations cleanly

// Default implementation for arithmetic types
template<typename T>
requires std::is_arithmetic_v<T>
struct cpo_impl<T> {
    static std::string call(T value) {
        return std::to_string(value);
    }
    
    // Traits can expose additional metadata
    static constexpr bool is_binary_serializable = true;
    static constexpr size_t max_size = 32;
};

// Specialized implementation for containers
template<typename T>
requires requires { typename T::value_type; T{}.begin(); T{}.end(); }
struct cpo_impl<T> {
    static std::string call(const T& container) {
        std::string result = "[";
        bool first = true;
        for (const auto& item : container) {
            if (!first) result += ",";
            // Recursive serialization using the same CPO
            result += serialize(item);  // This works because of tag_invoke!
            first = false;
        }
        result += "]";
        return result;
    }
    
    static constexpr bool is_binary_serializable = 
        cpo_impl<typename T::value_type>::is_binary_serializable;
    static constexpr size_t max_size = 1024;
};

// Complex implementation for strings (escaping, etc.)
template<>
struct cpo_impl<std::string> {
    static std::string call(const std::string& s) {
        std::string result = "\"";
        for (char c : s) {
            if (c == '"') result += "\\\"";
            else if (c == '\\') result += "\\\\";
            else if (c == '\n') result += "\\n";
            else result += c;
        }
        result += "\"";
        return result;
    }
    
    static constexpr bool is_binary_serializable = false;
    static constexpr size_t max_size = 2048;
};

} // namespace framework

// The tag_invoke interface - simple forwarding
template<typename T>
auto tag_invoke(framework::serialize_ftor, const T& value) 
    -> decltype(framework::cpo_impl<T>::call(value)) {
    return framework::cpo_impl<T>::call(value);
}

// Now we can use the traits for metaprogramming!
template<typename T>
constexpr bool is_serializable_v = requires {
    framework::cpo_impl<T>::call(std::declval<T>());
};

template<typename T>
constexpr bool is_binary_serializable_v = 
    is_serializable_v<T> && framework::cpo_impl<T>::is_binary_serializable;

template<typename T>
constexpr size_t max_serialize_size_v = 
    is_serializable_v<T> ? framework::cpo_impl<T>::max_size : 0;

// Compile-time algorithms using traits
template<typename... Ts>
constexpr bool all_binary_serializable_v = (is_binary_serializable_v<Ts> && ...);

template<typename... Ts>
constexpr size_t total_max_size_v = (max_serialize_size_v<Ts> + ...);

// Usage examples
int main() {
    using namespace framework;
    
    // Basic usage - same as simple tag_invoke approach
    std::cout << serialize(42) << "\n";           // "42"
    std::cout << serialize(3.14) << "\n";         // "3.140000"
    std::cout << serialize(std::string("hello \"world\"")) << "\n"; // "hello \"world\""
    
    std::vector<int> vec{1, 2, 3};
    std::cout << serialize(vec) << "\n";          // "[1,2,3]"
    
    // Advantages of traits approach:
    
    // 1. Compile-time queries
    static_assert(is_serializable_v<int>);
    static_assert(is_serializable_v<std::vector<int>>);
    static_assert(!is_serializable_v<void*>);  // Not supported
    
    // 2. Metadata access
    static_assert(is_binary_serializable_v<int>);
    static_assert(!is_binary_serializable_v<std::string>);
    static_assert(max_serialize_size_v<int> == 32);
    
    // 3. Template metaprogramming
    static_assert(all_binary_serializable_v<int, double, float>);
    static_assert(!all_binary_serializable_v<int, std::string>);
    
    // 4. Buffer sizing at compile time
    constexpr size_t buffer_size = total_max_size_v<int, double, float>;
    char buffer[buffer_size];  // Sized at compile time!
    
    std::cout << "Compile-time buffer size: " << buffer_size << "\n";
    
    return 0;
}