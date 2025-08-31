# RealVectorFramework

Header-only example framework built on TInCuP customization points.

- Docs: see `docs/CPO_INTEGRATION_GUIDE.md` for guidance on ADL `tag_invoke` vs `tincup::cpo_impl` trait specializations and when to use each.

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

