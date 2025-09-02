/**
RealVectorFramework - A Generic Library for Vector Operations and Algorithms

Copyright (c) National Technology & Engineering Solutions of Sandia,
LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
Government retains certain rights in this software.

Questions? Contact Greg von Winckel (gvonwin@sandia.gov)
*/

#pragma once

#include <concepts>                                                                                                     
#include <type_traits>                                                                                                  
#include <utility>                                                                                                      
 
namespace rvf { 

template<typename T>
concept nullable_pointer_c = requires(T t) {
  { static_cast<bool>(t) } -> std::convertible_to<bool>;  // contextually convertible to bool
  *t;  // must be dereferenceable
};

    
template<typename T>
  requires nullable_pointer_c<std::remove_reference_t<T>>
auto deref_if_needed(T&& x) noexcept(noexcept(*std::forward<T>(x)))    
-> decltype(*std::forward<T>(x)) {    
  return *std::forward<T>(x);    
}    
                                                                              
template<typename T>
  requires (!nullable_pointer_c<std::remove_reference_t<T>>)
auto deref_if_needed(T&& x) noexcept(std::is_nothrow_move_constructible_v<T>)                                          
-> T&& {                                                                                                               
  return std::forward<T>(x);                                                                                           
}                                                                                                                      
                                                                                                                         
template<typename V>
using deref_t = decltype( deref_if_needed( std::declval<V>() ) );  

} // namespace rvf
