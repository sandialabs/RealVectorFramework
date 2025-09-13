#pragma once

#include "core/rvf.hpp"
#include "core/type_support/std_cpo_impl.hpp"
#include "operations/advanced/self_map.hpp"
#include "operations/advanced/unary_in_place.hpp"
#include "operations/advanced/binary_in_place.hpp"
#include "operations/advanced/variadic_in_place.hpp"
#include "operations/advanced/fill.hpp"
#include "operations/advanced/relu.hpp"
#include "operations/advanced/softmax.hpp"
#include "operations/advanced/layer_norm.hpp"
#include "operations/advanced/matvec.hpp"
#include "operations/core/deref_if_needed.hpp"
#include "core/type_support/cmath/std_sqrt.hpp"
#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <ranges>
#include <type_traits>

namespace seqtrf {

using namespace rvf;

template<rvf::real_vector_c Vector>
using Sequence = std::vector<Vector>;

template<rvf::real_vector_c Vector>
using Matrix = std::vector<Vector>;

template<rvf::real_vector_c Vector>
using ScalarT = rvf::inner_product_return_t<Vector>;

// Compose two self-map operations (generic lambda; type deduced at call-site)
template<typename Op1, typename Op2>
auto compose(Op1 op1, Op2 op2) {
  return [op1 = std::move(op1), op2 = std::move(op2)](auto& y, const auto& x) {
    auto tmp = x;
    op1(tmp, x);
    op2(y, tmp);
  };
}

// Residual wrapper
template<typename Op>
auto add_residual(Op op) {
  return [op = std::move(op)](auto& y, const auto& x) {
    op(y, x);
    // For sequences: elementwise add per token
    if constexpr (std::ranges::range<std::decay_t<decltype(y)>>) {
      auto iy = std::ranges::begin(y);
      auto ix = std::ranges::begin(x);
      auto ey = std::ranges::end(y);
      for (; iy != ey && ix != std::ranges::end(x); ++iy, ++ix) {
        add_in_place(*iy, *ix);
      }
    }
  };
}

// Per-token layer norm wrapper for sequences
template<rvf::real_vector_c Vector, typename Op>
auto add_layer_norm(Op op, ScalarT<Vector> eps = static_cast<ScalarT<Vector>>(1e-5)) {
  return [op = std::move(op), eps](Sequence<Vector>& y, const Sequence<Vector>& x) {
    op(y, x);
    for (auto& tok : y) layer_norm(tok, eps);
  };
}

// Linear transform per token: y_i = W * x_i + b
template<rvf::real_vector_c Vector>
auto make_token_linear(const Matrix<Vector>& W, const Vector& b) {
  return [W, b](Sequence<Vector>& y, const Sequence<Vector>& x) {
    y = x; // shape copy (std::vector copy ok)
    for (std::size_t i = 0; i < y.size(); ++i) {
      matvec(y[i], W, x[i]);
      add_in_place(y[i], b);
    }
  };
}

// Feed-forward per token: W2 * relu(W1 * x + b1) + b2
template<rvf::real_vector_c Vector>
auto make_token_ffn(std::size_t d_model, std::size_t d_ff) {
  using Scalar = ScalarT<Vector>;
  Matrix<Vector> W1(d_ff, Vector(d_model));
  Vector b1(d_ff, Scalar(0));
  Matrix<Vector> W2(d_model, Vector(d_ff));
  Vector b2(d_model, Scalar(0));

  std::random_device rd; std::mt19937 gen(rd());
  auto init_matrix = [&](Matrix<Vector>& W, std::size_t in_dim, std::size_t out_dim) {
    Scalar std_dev = rvf::sqrt(Scalar(2) / Scalar(in_dim + out_dim));
    std::normal_distribution<std::remove_cvref_t<Scalar>> dist(Scalar(0), std_dev);
    auto rnd = [&](auto){ return dist(gen); };
    for (auto& row : W) unary_in_place(row, rnd);
  };
  init_matrix(W1, d_model, d_ff);
  init_matrix(W2, d_ff, d_model);

  auto lin1 = make_token_linear<Vector>(W1, b1);
  auto act = [](Sequence<Vector>& y, const Sequence<Vector>& x) {
    y = x;
    for (auto& tok : y) relu(tok);
  };
  auto lin2 = make_token_linear<Vector>(W2, b2);
  return compose(compose(lin1, act), lin2);
}

// Scaled dot-product attention over sequences (single head)
// Input/Output: Sequence<Vector> of length L
template<rvf::real_vector_c Vector>
auto make_sequence_attention(std::size_t d_model) {
  using Scalar = ScalarT<Vector>;
  Matrix<Vector> Wq(d_model, Vector(d_model));
  Matrix<Vector> Wk(d_model, Vector(d_model));
  Matrix<Vector> Wv(d_model, Vector(d_model));
  Matrix<Vector> Wo(d_model, Vector(d_model));

  std::random_device rd; std::mt19937 gen(rd());
  auto init_matrix = [&](Matrix<Vector>& W) {
    Scalar std_dev = rvf::sqrt(Scalar(2) / Scalar(2 * d_model));
    std::normal_distribution<std::remove_cvref_t<Scalar>> dist(Scalar(0), std_dev);
    auto rnd = [&](auto){ return dist(gen); };
    for (auto& row : W) unary_in_place(row, rnd);
  };
  init_matrix(Wq); init_matrix(Wk); init_matrix(Wv); init_matrix(Wo);

  const Scalar scale = Scalar(1) / rvf::sqrt(static_cast<Scalar>(d_model));

  return [Wq, Wk, Wv, Wo, scale](Sequence<Vector>& y, const Sequence<Vector>& x) {
    const std::size_t L = x.size();
    y = x; // shape

    // Precompute K,V projections
    Sequence<Vector> K = x;
    Sequence<Vector> V = x;
    for (std::size_t j = 0; j < L; ++j) {
      matvec(K[j], Wk, x[j]);
      matvec(V[j], Wv, x[j]);
    }

    // For each query token, compute attention over keys
    for (std::size_t i = 0; i < L; ++i) {
      Vector qi = x[i];
      matvec(qi, Wq, x[i]);

      // scores over j
      std::vector<Scalar> scores(L, Scalar(0));
      for (std::size_t j = 0; j < L; ++j) {
        scores[j] = scale * inner_product(qi, K[j]);
      }
      softmax(scores);

      // weighted sum of V
      Vector out = V[0];
      rvf::fill(out, Scalar(0));
      for (std::size_t j = 0; j < L; ++j) {
        auto vh = rvf::clone(V[j]); auto& vj = rvf::deref_if_needed(vh);
        scale_in_place(vj, scores[j]);
        add_in_place(out, vj);
      }

      // output projection
      matvec(y[i], Wo, out);
    }
  };
}

// Transformer block over sequences: Attn + Residual + LN, then FFN + Residual + LN
template<rvf::real_vector_c Vector>
auto make_sequence_block(std::size_t d_model, std::size_t d_ff) {
  auto attn = make_sequence_attention<Vector>(d_model);
  auto ffn = make_token_ffn<Vector>(d_model, d_ff);
  return compose(
    add_layer_norm<Vector>(add_residual(std::move(attn))),
    add_layer_norm<Vector>(add_residual(std::move(ffn)))
  );
}

// Stack N blocks
template<rvf::real_vector_c Vector>
auto make_sequence_transformer(std::size_t d_model, std::size_t n_layers, std::size_t d_ff) {
  std::function<void(Sequence<Vector>&, const Sequence<Vector>&)> tr =
    [](Sequence<Vector>& y, const Sequence<Vector>& x){ y = x; };
  for (std::size_t i = 0; i < n_layers; ++i) {
    auto blk = make_sequence_block<Vector>(d_model, d_ff);
    tr = compose(std::move(tr), std::move(blk));
  }
  return tr;
}

} // namespace seqtrf
