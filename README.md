# RealVectorFramework

Header-only example framework built on TInCuP customization points.

- Docs: see `docs/CPO_INTEGRATION_GUIDE.md` for guidance on ADL `tag_invoke` vs `tincup::cpo_impl` trait specializations and when to use each.

## Cloning Idiom (Owner + Reference)

Many RVF operations accept and return either values or smart-pointer-like wrappers. The `clone` CPO may therefore return either a value (e.g., `std::vector<double>`) or a wrapper (e.g., `std::unique_ptr<std::vector<double>>`). Use `deref_if_needed` to obtain a reference to the underlying vector.

Recommended idiom: keep the owner and the reference adjacent to avoid lifetime issues and to satisfy concept-constrained APIs (which check types before applying conversions).

```cpp
auto cl = rvf::clone(x); auto& xr = rvf::deref_if_needed(cl);
// use xr as a regular vector; cl owns storage so xr stays valid
```

Rationale:
- Concept checks run before implicit conversions, so passing temporary wrappers to functions constrained on `real_vector_c` will fail. Binding a reference (`xr`) avoids this.
- Keeping the owner (`cl`) and the reference (`xr`) adjacent makes the code read more mathematically while preventing dangling references.

This idiom is used throughout the algorithms in this repo (e.g., Conjugate Gradient, Sherman–Morrison, gradient descent with bounds).

## Examples

- `examples/main.cpp`: Uses the generic, ranges-based `tag_invoke` implementations in `rvf`.
- `examples/traits_vs_ranges.cpp`: Includes an optional header `operations/std_cpo_impl.hpp` to illustrate routing `std::vector` operations through `tincup::cpo_impl` trait specializations.

Build:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run examples:

```
./build/examples/example
./build/examples/example_traits
```
