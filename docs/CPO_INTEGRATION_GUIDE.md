## RVF CPO Integration: ADL vs Trait Specializations

This guide explains two complementary ways to provide implementations for RVF customization points (CPOs), when to use each, and how to combine them effectively.

### Approaches

1) ADL `tag_invoke` overloads (generic, ranges-based)
- Location: `rvf` namespace, e.g. `include/operations/std_ranges_support.hpp`.
- Idea: Provide a single set of `tag_invoke` templates constrained on concepts (e.g., `std::ranges::range`) that work for many containers (e.g., `std::vector`, `std::array`, views/spans).
- Pros:
  - Broad coverage with minimal code.
  - No per-type boilerplate; great defaults for standard/ranges types.
  - Keeps call sites simple (just call `rvf::add_in_place(y, x)`, etc.).
- Cons:
  - Less per-type control/optimization.
  - Harder to plug in specialized backends without replacing the generic overloads.

2) `tincup::cpo_impl` trait specializations (formatter-style)
- Location: `namespace tincup`; provide `struct cpo_impl<CPO, Target> { static auto call(...); }`.
- Optionally add an ADL-visible shim in `rvf` that forwards to the trait.
- Pros:
  - Excellent for third-party types you don’t control (Kokkos, Eigen, device vectors); no foreign namespace edits.
  - Clear place for optimized fast paths for specific targets.
  - Explicit and stable integration boundary; easy to test and version.
- Cons:
  - Per-type boilerplate; you must declare a specialization for each target family.
  - More moving parts (trait + optional shim).

### Recommended Strategy

- Default: Keep the generic ranges-based `rvf` `tag_invoke` overloads as broad, ergonomic coverage.
- Extend: Add `tincup::cpo_impl` specializations for third-party types or to optimize hot paths.
- Hybrid pattern (trait-first, generic fallback):
  - Provide an ADL `tag_invoke` shim in `rvf` that forwards to `tincup::cpo_impl<CPO, R>::call(...)` when available.
  - Provide a second constrained overload that implements the generic ranges behavior when no trait specialization exists.
  - Result: consumers get optimized implementations automatically where provided, and generic behavior otherwise.

Note: TInCuP exposes detection helpers to write these constraints safely:
- `tincup::has_cpo_impl_for_c<CPO, Target, Args...>` and
- `tincup::has_specialized_cpo_impl_c<CPO, Args...>`.

When constraining `tag_invoke` overloads for the same CPO, prefer `has_cpo_impl_for_c` with an explicit target type to avoid recursive constraints.

### Example: Trait + Shim for std::vector (Illustration)

- See `include/operations/std_cpo_impl.hpp` for a concrete example:
  - Specializes `tincup::cpo_impl` for `std::vector<T, Alloc>` for `add_in_place`, `scale_in_place`, `inner_product`, `dimension`, `clone`.
  - Adds `rvf::tag_invoke` shims that forward to those traits.
  - This header is optional; include it from a TU to activate trait-based routing for `std::vector`.

### When to Choose Which

- Use generic ADL `tag_invoke` in `rvf` when:
  - You want broad coverage for standard/ranges-friendly containers.
  - Maintainability and minimal boilerplate are more important than per-type tuning.

- Use `tincup::cpo_impl` specializations when:
  - Integrating types you don’t own (no editing their namespaces).
  - You need optimized implementations (SIMD, device memory, custom allocators/policies).
  - You want clean separation of “what RVF does generically” from “what we do for this backend.”

- Use both (hybrid) when:
  - You want generic behavior by default but allow seamless drop-in optimized backends.

### Using the cpo-generator

TInCuP’s `cpo-generator` can emit skeletons for both the trait specialization and the ADL shim. Typical flows:

- Emit trait-only for a third-party type:
  - `--trait-impl-only --impl-target 'Kokkos::View<$T, $Rest...>'`

- Emit trait + ADL shim (in `rvf`) for a standard type:
  - `--emit-trait-impl --emit-adl-shim --shim-namespace 'rvf' --impl-target 'std::vector<$T, $Alloc>'`

See TInCuP’s README for detailed flags and templates.

### Performance & Testing

- Benchmark trait specializations vs the generic ranges implementation in your target environment.
- Ensure concept checks still hold (e.g., `real_vector_c`) for your types and specializations.
- Add unit tests that validate both generic and specialized paths where applicable.

### Takeaway

- Keep the generic, concept-constrained `rvf` `tag_invoke` overloads for broad usability.
- Layer `tincup::cpo_impl` specializations for backends and optimizations.
- Prefer the hybrid pattern to get the best of both worlds without changing call sites.
