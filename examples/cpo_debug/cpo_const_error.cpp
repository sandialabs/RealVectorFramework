#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include "core/rvf.hpp"

// 1. Define the minimal CPO
namespace debug {
  inline constexpr struct debug_axpy_ftor final : tincup::cpo_base<debug_axpy_ftor> {
    TINCUP_CPO_TAG("debug_axpy")
    using tincup::cpo_base<debug_axpy_ftor>::operator();
  } debug_axpy;
}

// 2. Provide the implementation via the correct cpo_impl specialization pattern
namespace tincup {
  template<typename T, typename Alloc>
  struct cpo_impl<debug::debug_axpy_ftor, std::vector<T, Alloc>> {
    // The implementation requires the 3rd argument to be const
    static void call(std::vector<T, Alloc>& y, T alpha, const std::vector<T, Alloc>& x) {
      std::cout << "tag_invoke for debug_axpy called successfully!" << std::endl;
      for (size_t i = 0; i < y.size(); ++i) {
        y[i] += alpha * x[i];
      }
    }
  };
}

// 3. Provide the ADL-visible tag_invoke shim that forwards to the implementation
namespace debug {
  template<typename T, typename Alloc>
  constexpr auto tag_invoke(debug_axpy_ftor,
               std::vector<T, Alloc>& y,
               T alpha,
               const std::vector<T, Alloc>& x) {
    return tincup::cpo_impl<debug_axpy_ftor, std::vector<T, Alloc>>::call(y, alpha, x);
  }
}

int main() {
  std::vector<double> y = {1.0, 2.0, 3.0};
  std::vector<double> x = {4.0, 5.0, 6.0};
  double alpha = 2.0;

  std::cout << "Attempting to call debug_axpy with a non-const vector..." << std::endl;

  // 4. This call will fail to compile.
  // The CPO system will search for a tag_invoke matching (CPO, vector&, double, vector&).
  // The only candidate is the shim in namespace debug, but its signature requires (..., const vector&).
  // Because the signatures do not match exactly, the CPO lookup fails.
  // The goal for the next session is to improve the static_assert in TInCuP
  // to detect this near-miss and provide a better error message.
  debug::debug_axpy(y, alpha, x);

  // This code will not be reached
  std::cout << "Final y vector: [" << y[0] << ", " << y[1] << ", " << y[2] << "]" << std::endl;

  return 0;
}
