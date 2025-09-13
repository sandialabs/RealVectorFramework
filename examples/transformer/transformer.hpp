#pragma once

#include "core/rvf.hpp"
#include "core/type_support/std_cpo_impl.hpp"
#include "operations/advanced/self_map.hpp"
#include "operations/advanced/relu.hpp"
#include "operations/advanced/softmax.hpp" 
#include "operations/advanced/layer_norm.hpp"
#include "operations/advanced/matvec.hpp"
#include "core/type_support/cmath/std_abs.hpp"
#include "core/type_support/cmath/std_exp.hpp"
#include "core/type_support/cmath/std_fmax.hpp"
#include "core/type_support/cmath/std_fmin.hpp"
#include <vector>
#include <random>
#include <functional>
#include <cmath>

namespace transformer {

using namespace rvf;
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

/**
 * @brief Generic linear transformation as a simple callable
 * 
 * This is much simpler than the previous class-based approach.
 * It's just a function that captures weights and uses RVF CPOs.
 */
auto make_linear_transform(const Matrix& weights, const Vector& bias) {
  return [weights, bias](Vector& y, const Vector& x) {
  matvec(y, weights, x);    // RVF's native matvec
  add_in_place(y, bias);    // RVF's native add_in_place
  };
}

/**
 * @brief Generic activation function wrapper
 * 
 * This wraps any RVF CPO into a self_map_c operator.
 * Much more general than specific activation classes!
 */
auto make_activation(tincup::cpo_c auto activation_cpo) {
  return [activation_cpo](Vector& y, const Vector& x) {
  y = x;        // Copy input
  activation_cpo(y);  // Apply activation using RVF CPO
  };
}

/**
 * @brief Composed operation wrapper that satisfies self_map_c
 */
template<typename Op1, typename Op2>
class ComposedOp {
public:
  ComposedOp(Op1&& op1, Op2&& op2) 
    : op1_(std::move(op1)), op2_(std::move(op2)) {}
  
  void operator()(Vector& y, const Vector& x) const {
    Vector temp(x.size());
    op1_(temp, x);    // Apply first operation
    op2_(y, temp);    // Apply second operation  
  }
  
private:
  mutable Op1 op1_;
  mutable Op2 op2_;
};

/**
 * @brief Compose two self_map_c operators sequentially
 * 
 * This is the key insight: we can compose any self_map_c operators!
 * This makes building complex architectures trivial.
 */
template<typename Op1, typename Op2>
auto compose(Op1&& op1, Op2&& op2) {
  return ComposedOp(std::move(op1), std::move(op2));
}

/**
 * @brief Residual operation wrapper that satisfies self_map_c
 */
template<typename Op>
class ResidualOp {
public:
  explicit ResidualOp(Op&& op) : op_(std::move(op)) {}
  
  void operator()(Vector& y, const Vector& x) const {
    op_(y, x);  // Apply operation
    add_in_place(y, x);      // Add residual using RVF CPO
  }
  
private:
  mutable Op op_;
};

/**
 * @brief Add residual connection to any self_map_c operator
 * 
 * Residual(f)(x) = f(x) + x
 * This works with ANY self_map_c operator!
 */
template<typename Op>
auto add_residual(Op&& op) {
  return ResidualOp(std::move(op));	
//  return ResidualOp<std::decay_t<Op>>(std::forward<Op>(op));
}

/**
 * @brief Layer normalization operation wrapper that satisfies self_map_c
 */
template<typename Op>
class LayerNormOp {
public:
  explicit LayerNormOp(Op&& op, double eps = 1e-5) 
    : op_(std::move(op)), eps_(eps) {}
  
  void operator()(Vector& y, const Vector& x) const {
    op_(y, x);  // Apply operation
    layer_norm(y, eps_);     // Normalize using RVF CPO
  }
  
private:
  mutable Op op_;
  double eps_;
};

/**
 * @brief Add layer normalization to any self_map_c operator
 * 
 * LayerNorm(f)(x) = layer_norm(f(x))
 * Again, works with ANY self_map_c operator!
 */
template<typename Op>
auto add_layer_norm(Op&& op, double eps = 1e-5) {
  return LayerNormOp(std::move(op), eps);
}

// ============================================================================
// SIMPLIFIED TRANSFORMER COMPONENTS
// ============================================================================

/**
 * @brief Create feed-forward network using functional composition
 * 
 * This is MUCH simpler than the class-based approach!
 */
auto make_feed_forward(std::size_t d_model, std::size_t d_ff) {
  // Initialize weights randomly
  Matrix W1(d_ff, Vector(d_model));
  Vector b1(d_ff, 0.0);
  Matrix W2(d_model, Vector(d_ff));  
  Vector b2(d_model, 0.0);
  
  // Xavier initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  auto init_matrix = [&](Matrix& W, std::size_t in_dim, std::size_t out_dim) {
  double std_dev = std::sqrt(2.0 / (in_dim + out_dim));
  std::normal_distribution<> dist(0.0, std_dev);
  auto randomize = [&](auto) { return dist(gen); };
  for (auto& row : W) {
    unary_in_place(row,randomize);
  }
  };
  
  init_matrix(W1, d_model, d_ff);
  init_matrix(W2, d_ff, d_model);
  
  // Compose operations using RVF CPOs:
  // FFN(x) = W2 * ReLU(W1 * x + b1) + b2
  return compose(
  make_linear_transform(W1, b1),   // First linear layer
  compose(
    make_activation(relu),         // ReLU activation 
    make_linear_transform(W2, b2)  // Second linear layer
  )
  );
}

// ============================================================================
// MODULAR ATTENTION SYSTEM
// ============================================================================

/**
 * @brief Sequence representation for multi-token attention
 * 
 * A sequence is a matrix where each row is a token embedding
 */
using Sequence = std::vector<Vector>;

/**
 * @brief Modular attention scoring mechanisms
 * 
 * Each scorer computes attention weights between query and key vectors
 */
namespace attention_scorers {

// Scaled dot-product attention scoring
class ScaledDotProduct {
public:
  explicit ScaledDotProduct(std::size_t d_k) : d_k_(d_k), scale_(1.0 / std::sqrt(static_cast<double>(d_k))) {}
  
  double operator()(const Vector& query, const Vector& key) const {
    return inner_product(query, key) * scale_;
  }
  
private:
  std::size_t d_k_;
  double scale_;
};

// Additive attention scoring (Bahdanau-style)
class Additive {
public:
  Additive(std::size_t d_k, std::size_t d_hidden = 128) : d_k_(d_k), d_hidden_(d_hidden) {
    // Initialize weight matrices for additive attention
    W_q_ = Matrix(d_hidden, Vector(d_k));
    W_k_ = Matrix(d_hidden, Vector(d_k));
    v_ = Vector(d_hidden);
    
    // Xavier initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    auto init_matrix = [&](Matrix& W, std::size_t in_dim, std::size_t out_dim) {
      double std_dev = std::sqrt(2.0 / (in_dim + out_dim));
      std::normal_distribution<> dist(0.0, std_dev);
      auto randomize = [&](auto) { return dist(gen); };
      for(auto& row : W) {
        unary_in_place(row, randomize);
      }
    };
    
    init_matrix(W_q_, d_k, d_hidden);
    init_matrix(W_k_, d_k, d_hidden);
    
    std::normal_distribution<> dist(0.0, std::sqrt(2.0 / d_hidden));
    auto randomize = [&](auto) { return dist(gen); };
    unary_in_place(v_, randomize);
  }
  
  double operator()(const Vector& query, const Vector& key) const {
    Vector h_q(d_hidden_), h_k(d_hidden_), h_sum(d_hidden_);
    
    // h_q = W_q * query, h_k = W_k * key
    matvec(h_q, W_q_, query);
    matvec(h_k, W_k_, key);
    
    // h_sum = tanh(h_q + h_k)
    h_sum = rvf::clone(h_q);
    add_in_place(h_sum, h_k);
    unary_in_place(h_sum, [](double x) { return std::tanh(x); });
    
    // score = v^T * h_sum
    return inner_product(v_, h_sum);
  }
  
private:
  std::size_t d_k_, d_hidden_;
  Matrix W_q_, W_k_;
  Vector v_;
};

// Multiplicative attention (Luong-style)
class Multiplicative {
public:
  explicit Multiplicative(std::size_t d_k) : d_k_(d_k) {
    // Initialize weight matrix
    W_ = Matrix(d_k, Vector(d_k));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    double std_dev = std::sqrt(2.0 / (2 * d_k));
    std::normal_distribution<> dist(0.0, std_dev);
    auto randomize = [&](auto) { return dist(gen); };
    
    for(auto& row : W_) {
      unary_in_place(row, randomize);
    }
  }
  
  double operator()(const Vector& query, const Vector& key) const {
    Vector transformed_query(d_k_);
    matvec(transformed_query, W_, query);
    return inner_product(transformed_query, key);
  }
  
private:
  std::size_t d_k_;
  Matrix W_;
};

} // namespace attention_scorers

/**
 * @brief Modular attention normalization mechanisms
 */
namespace attention_normalizers {

// Standard softmax normalization
struct Softmax {
  void operator()(Vector& scores) const {
    softmax(scores);  // Use RVF's native softmax
  }
};

// Sparsemax normalization (creates sparse attention weights)
struct Sparsemax {
  void operator()(Vector& scores) const {
    // Simplified sparsemax implementation
    // Sort scores in descending order
    Vector sorted_scores = rvf::clone(scores);
    std::sort(sorted_scores.rbegin(), sorted_scores.rend());
    
    // Find threshold
    double sum = 0.0;
    std::size_t k = 1;
    for(std::size_t i = 0; i < sorted_scores.size(); ++i) {
      sum += sorted_scores[i];
      double threshold = (sum - 1.0) / static_cast<double>(i + 1);
      if(sorted_scores[i] > threshold) {
        k = i + 1;
      }
    }
    
    double tau = (std::accumulate(sorted_scores.begin(), sorted_scores.begin() + k, 0.0) - 1.0) / static_cast<double>(k);
    
    // Apply threshold
    unary_in_place(scores, [tau](double x) { return std::max(0.0, x - tau); });
  }
};

// Entmax normalization (configurable sparsity)
struct Entmax {
  explicit Entmax(double alpha = 1.5) : alpha_(alpha) {}
  
  void operator()(Vector& scores) const {
    // Simplified entmax implementation - falls back to softmax for alpha=1
    if(std::abs(alpha_ - 1.0) < 1e-6) {
      softmax(scores);
      return;
    }
    
    // For alpha != 1, use approximate entmax
    // This is a simplified version - full implementation would be more complex
    Vector exp_scores(scores.size());
    double alpha_minus_one = alpha_ - 1.0;
    
    unary_in_place(scores, [alpha_minus_one](double x) {
      return std::pow(std::max(0.0, 1.0 + alpha_minus_one * x), 1.0 / alpha_minus_one);
    });
    
    double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
    if(sum > 1e-10) {
      scale_in_place(scores, 1.0 / sum);
    }
  }
  
private:
  double alpha_;
};

} // namespace attention_normalizers

/**
 * @brief Generic multi-head attention with pluggable components
 */
template<typename Scorer, typename Normalizer>
class ModularMultiHeadAttention {
public:
  ModularMultiHeadAttention(std::size_t d_model, std::size_t n_heads, Scorer scorer, Normalizer normalizer)
    : d_model_(d_model), n_heads_(n_heads), d_k_(d_model / n_heads), 
      scorer_(std::move(scorer)), normalizer_(std::move(normalizer)) {
    
    // Initialize projection matrices
    W_q_ = Matrix(d_model, Vector(d_model));
    W_k_ = Matrix(d_model, Vector(d_model));
    W_v_ = Matrix(d_model, Vector(d_model));
    W_o_ = Matrix(d_model, Vector(d_model));
    
    init_weights();
  }
  
  // Single-vector attention (for compatibility) - mutable to work with composition
  void operator()(Vector& output, const Vector& input) {
    Sequence seq = {input};
    Sequence result_seq(1, Vector(d_model_));
    
    (*this)(result_seq, seq, seq, seq);  // Self-attention
    output = result_seq[0];
  }
  
  // Full sequence-to-sequence attention - mutable to work with composition
  void operator()(Sequence& output, const Sequence& query_seq, const Sequence& key_seq, const Sequence& value_seq) {
    std::size_t seq_len = query_seq.size();
    output.resize(seq_len, Vector(d_model_));
    
    for(std::size_t i = 0; i < seq_len; ++i) {
      attend_token(output[i], query_seq[i], key_seq, value_seq);
    }
  }
  
private:
  void init_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    auto init_matrix = [&](Matrix& W, std::size_t in_dim, std::size_t out_dim) {
      double std_dev = std::sqrt(2.0 / (in_dim + out_dim));
      std::normal_distribution<> dist(0.0, std_dev);
      auto randomize = [&](auto) { return dist(gen); };
      for(auto& row : W) {
        unary_in_place(row, randomize);
      }
    };
    
    init_matrix(W_q_, d_model_, d_model_);
    init_matrix(W_k_, d_model_, d_model_);
    init_matrix(W_v_, d_model_, d_model_);
    init_matrix(W_o_, d_model_, d_model_);
  }
  
  void attend_token(Vector& output, const Vector& query_token, const Sequence& key_seq, const Sequence& value_seq) {
    // Project query
    Vector Q(d_model_);
    matvec(Q, W_q_, query_token);
    
    // Multi-head attention
    Vector concatenated_heads(d_model_, 0.0);
    
    for(std::size_t h = 0; h < n_heads_; ++h) {
      Vector head_output = compute_single_head(Q, key_seq, value_seq, h);
      
      // Copy head output to concatenated result
      std::size_t start_idx = h * d_k_;
      for(std::size_t i = 0; i < d_k_ && start_idx + i < d_model_; ++i) {
        concatenated_heads[start_idx + i] = head_output[i];
      }
    }
    
    // Output projection
    matvec(output, W_o_, concatenated_heads);
  }
  
  Vector compute_single_head(const Vector& Q, const Sequence& key_seq, const Sequence& value_seq, std::size_t head_idx) {
    std::size_t seq_len = key_seq.size();
    Vector scores(seq_len);
    
    // Extract head-specific query
    Vector Q_head = extract_head(Q, head_idx);
    
    // Compute attention scores
    for(std::size_t i = 0; i < seq_len; ++i) {
      Vector K_i(d_model_);
      matvec(K_i, W_k_, key_seq[i]);
      Vector K_head = extract_head(K_i, head_idx);
      
      scores[i] = scorer_(Q_head, K_head);
    }
    
    // Normalize scores
    normalizer_(scores);
    
    // Compute weighted sum of values
    Vector result(d_k_, 0.0);
    for(std::size_t i = 0; i < seq_len; ++i) {
      Vector V_i(d_model_);
      matvec(V_i, W_v_, value_seq[i]);
      Vector V_head = extract_head(V_i, head_idx);
      
      Vector weighted_V = rvf::clone(V_head);
      scale_in_place(weighted_V, scores[i]);
      add_in_place(result, weighted_V);
    }
    
    return result;
  }
  
  Vector extract_head(const Vector& full_vector, std::size_t head_idx) {
    std::size_t start_idx = head_idx * d_k_;
    Vector head(d_k_);
    
    for(std::size_t i = 0; i < d_k_ && start_idx + i < full_vector.size(); ++i) {
      head[i] = full_vector[start_idx + i];
    }
    
    return head;
  }
  
  std::size_t d_model_, n_heads_, d_k_;
  Matrix W_q_, W_k_, W_v_, W_o_;
  Scorer scorer_;
  Normalizer normalizer_;
};

/**
 * @brief Factory functions for different attention variants
 */

// Standard scaled dot-product attention
auto make_scaled_dot_product_attention(std::size_t d_model, std::size_t n_heads) {
  return ModularMultiHeadAttention(
    d_model, n_heads,
    attention_scorers::ScaledDotProduct(d_model / n_heads),
    attention_normalizers::Softmax{}
  );
}

// Sparse attention using sparsemax
auto make_sparse_attention(std::size_t d_model, std::size_t n_heads) {
  return ModularMultiHeadAttention(
    d_model, n_heads,
    attention_scorers::ScaledDotProduct(d_model / n_heads),
    attention_normalizers::Sparsemax{}
  );
}

// Additive attention (Bahdanau-style)
auto make_additive_attention(std::size_t d_model, std::size_t n_heads, std::size_t d_hidden = 128) {
  return ModularMultiHeadAttention(
    d_model, n_heads,
    attention_scorers::Additive(d_model / n_heads, d_hidden),
    attention_normalizers::Softmax{}
  );
}

// Multiplicative attention (Luong-style)
auto make_multiplicative_attention(std::size_t d_model, std::size_t n_heads) {
  return ModularMultiHeadAttention(
    d_model, n_heads,
    attention_scorers::Multiplicative(d_model / n_heads),
    attention_normalizers::Softmax{}
  );
}

// Entmax attention with configurable sparsity
auto make_entmax_attention(std::size_t d_model, std::size_t n_heads, double alpha = 1.5) {
  return ModularMultiHeadAttention(
    d_model, n_heads,
    attention_scorers::ScaledDotProduct(d_model / n_heads),
    attention_normalizers::Entmax(alpha)
  );
}

/**
 * @brief Self-map wrapper for attention mechanisms
 * 
 * This wraps any attention mechanism to work with the self_map_c concept
 */
template<typename AttentionImpl>
auto make_attention_self_map(AttentionImpl&& attention) {
  return [attention = std::move(attention)](Vector& output, const Vector& input) mutable {
    attention(output, input);
  };
}

// Backward compatibility - default attention using scaled dot-product
auto make_attention(std::size_t d_model, std::size_t n_heads) {
  return make_attention_self_map(make_scaled_dot_product_attention(d_model, n_heads));
}

/**
 * @brief Create transformer block using functional composition
 * 
 * This is the key simplification: we build complex architectures
 * by composing simple functions that use RVF CPOs!
 */
auto make_transformer_block(std::size_t d_model, std::size_t n_heads, std::size_t d_ff) {
  auto attention = make_attention(d_model, n_heads);
  auto feed_forward = make_feed_forward(d_model, d_ff);
  
  // TransformerBlock(x) = LayerNorm(FFN(LayerNorm(Attention(x) + x)) + LayerNorm(Attention(x) + x))
  // This is much clearer than the class-based approach!
  return compose(
    add_layer_norm(add_residual(attention)),       // Self-attention + residual + layer norm
    add_layer_norm(add_residual(feed_forward))     // Feed-forward + residual + layer norm  
  );
}

/**
 * @brief Create complete transformer using functional composition
 * 
 * This demonstrates the power of the functional approach:
 * building a complex model by composing simple operations!
 */
auto make_transformer( std::size_t d_model, 
		     std::size_t n_layers, 
		     std::size_t n_heads, 
		     std::size_t d_ff ) {
  // Start with identity transformation
  std::function<void(Vector&, const Vector&)> transformer = [](Vector& y, const Vector& x) { y = x; };
  
  // Compose multiple transformer blocks
  for(std::size_t i = 0; i < n_layers; ++i) {
  auto block = make_transformer_block(d_model, n_heads, d_ff);
  transformer = compose(std::move(transformer), std::move(block));
  }
  
  return transformer;
}

// ============================================================================
// ULTRA-GENERIC ARCHITECTURE BUILDER
// ============================================================================

/**
 * @brief Sequential operation wrapper that satisfies self_map_c
 */
template<typename... Ops>
class SequentialOp {
public:
  explicit SequentialOp(Ops&&... ops) : ops_(std::move(ops)...) {}
  
  void operator()(Vector& y, const Vector& x) const {
    Vector temp1 = x, temp2;
    Vector* current = &temp1;
    Vector* next = &temp2;
    
    // Apply operations sequentially
    std::apply([&](auto&... ops) {
      ((ops(*next, *current), std::swap(current, next)), ...);
    }, ops_);
    
    y = *current;
  }
  
private:
  mutable std::tuple<Ops...> ops_;
};

/**
 * @brief Parallel sum operation wrapper that satisfies self_map_c
 */
template<typename... Ops>
class ParallelSumOp {
public:
  explicit ParallelSumOp(Ops&&... ops) : ops_(std::move(ops)...) {}
  
  void operator()(Vector& y, const Vector& x) const {
    y.assign(x.size(), 0.0);  // Initialize to zero
    Vector temp(x.size());
    
    // Apply each operation and sum results
    std::apply([&](auto&... ops) {
      ((ops(temp, x), add_in_place(y, temp)), ...);
    }, ops_);
  }
  
private:
  mutable std::tuple<Ops...> ops_;
};

/**
 * @brief Generic sequential architecture builder
 * 
 * This allows building ANY sequential architecture using RVF operations!
 * Much more general than transformer-specific code.
 */
template<typename... Ops>
auto make_sequential(Ops&&... ops) {
  return SequentialOp(std::move(ops)...);
}

/**
 * @brief Generic parallel architecture builder  
 * 
 * This allows building ensemble models, multi-branch architectures, etc.
 */
template<typename... Ops>
auto make_parallel_sum(Ops&&... ops) {
  return ParallelSumOp(std::move(ops)...);
}

// ============================================================================
// ADVANCED TRAINING SYSTEM WITH RVF OPTIMIZATION
// ============================================================================

/**
 * @brief Parameter extraction interface
 * 
 * Allows flexible interpretation of parameter vectors for different model types
 */
template<typename Model>
class ParameterExtractor {
public:
  virtual ~ParameterExtractor() = default;
  virtual void apply_parameters(Model& model, const Vector& params) const = 0;
  virtual void extract_gradient(Vector& grad, const Model& model, const Vector& output_grad) const = 0;
  virtual std::size_t parameter_count() const = 0;
};

/**
 * @brief Simple bias-only parameter extractor (for backward compatibility)
 */
template<self_map_c<Vector> Model>
class BiasParameterExtractor : public ParameterExtractor<Model> {
public:
  explicit BiasParameterExtractor(std::size_t output_dim) : output_dim_(output_dim) {}
  
  void apply_parameters(Model& model, const Vector& params) const override {
    // Store bias for later application
    bias_.assign(params.begin(), params.begin() + std::min(output_dim_, params.size()));
    if (bias_.size() < output_dim_) bias_.resize(output_dim_, 0.0);
  }
  
  void extract_gradient(Vector& grad, const Model& model, const Vector& output_grad) const override {
    grad.resize(output_dim_);
    std::copy(output_grad.begin(), 
          output_grad.begin() + std::min(output_dim_, output_grad.size()), 
          grad.begin());
  }
  
  std::size_t parameter_count() const override { return output_dim_; }
  
  Vector get_bias() const { return bias_; }
  
private:
  std::size_t output_dim_;
  mutable Vector bias_;
};

/**
 * @brief Loss function interface
 */
class LossFunction {
public:
  virtual ~LossFunction() = default;
  virtual double compute_loss(const Vector& prediction, const Vector& target) const = 0;
  virtual void compute_gradient(Vector& grad, const Vector& prediction, const Vector& target) const = 0;
};

/**
 * @brief Loss function implementations
 */
namespace loss_functions {

class MeanSquaredError : public LossFunction {
public:
  double compute_loss(const Vector& prediction, const Vector& target) const override {
    Vector diff = rvf::clone(prediction);
    Vector neg_target = rvf::clone(target);
    scale_in_place(neg_target, -1.0);
    add_in_place(diff, neg_target);  // diff = prediction - target
    return 0.5 * inner_product(diff, diff);
  }
  
  void compute_gradient(Vector& grad, const Vector& prediction, const Vector& target) const override {
    grad = rvf::clone(prediction);
    Vector neg_target = rvf::clone(target);
    scale_in_place(neg_target, -1.0);
    add_in_place(grad, neg_target);  // grad = prediction - target
  }
};

class CrossEntropy : public LossFunction {
public:
  double compute_loss(const Vector& prediction, const Vector& target) const override {
    double loss = 0.0;
    for (std::size_t i = 0; i < std::min(prediction.size(), target.size()); ++i) {
      double p = std::max(1e-15, std::min(1.0 - 1e-15, prediction[i])); // Clip for stability
      loss -= target[i] * std::log(p);
    }
    return loss;
  }
  
  void compute_gradient(Vector& grad, const Vector& prediction, const Vector& target) const override {
    grad.resize(prediction.size());
    for (std::size_t i = 0; i < std::min(prediction.size(), target.size()); ++i) {
      double p = std::max(1e-15, std::min(1.0 - 1e-15, prediction[i])); // Clip for stability
      grad[i] = -target[i] / p;
    }
  }
};

class HuberLoss : public LossFunction {
public:
  explicit HuberLoss(double delta = 1.0) : delta_(delta) {}
  
  double compute_loss(const Vector& prediction, const Vector& target) const override {
    double loss = 0.0;
    for (std::size_t i = 0; i < std::min(prediction.size(), target.size()); ++i) {
      double diff = prediction[i] - target[i];
      double abs_diff = std::abs(diff);
      if (abs_diff <= delta_) {
        loss += 0.5 * diff * diff;
      } else {
        loss += delta_ * abs_diff - 0.5 * delta_ * delta_;
      }
    }
    return loss;
  }
  
  void compute_gradient(Vector& grad, const Vector& prediction, const Vector& target) const override {
    grad.resize(prediction.size());
    for (std::size_t i = 0; i < std::min(prediction.size(), target.size()); ++i) {
      double diff = prediction[i] - target[i];
      if (std::abs(diff) <= delta_) {
        grad[i] = diff;
      } else {
        grad[i] = delta_ * (diff > 0 ? 1.0 : -1.0);
      }
    }
  }
  
private:
  double delta_;
};

} // namespace loss_functions

/**
 * @brief Regularization interface
 */
class Regularizer {
public:
  virtual ~Regularizer() = default;
  virtual double compute_penalty(const Vector& params) const = 0;
  virtual void add_gradient(Vector& grad, const Vector& params) const = 0;
};

/**
 * @brief Regularization implementations
 */
namespace regularizers {

class L2Regularizer : public Regularizer {
public:
  explicit L2Regularizer(double lambda = 0.01) : lambda_(lambda) {}
  
  double compute_penalty(const Vector& params) const override {
    return 0.5 * lambda_ * inner_product(params, params);
  }
  
  void add_gradient(Vector& grad, const Vector& params) const override {
    Vector reg_grad = rvf::clone(params);
    scale_in_place(reg_grad, lambda_);
    add_in_place(grad, reg_grad);
  }
  
private:
  double lambda_;
};

class L1Regularizer : public Regularizer {
public:
  explicit L1Regularizer(double lambda = 0.01) : lambda_(lambda) {}
  
  double compute_penalty(const Vector& params) const override {
    double penalty = 0.0;
    for (double p : params) {
      penalty += std::abs(p);
    }
    return lambda_ * penalty;
  }
  
  void add_gradient(Vector& grad, const Vector& params) const override {
    for (std::size_t i = 0; i < std::min(grad.size(), params.size()); ++i) {
      grad[i] += lambda_ * (params[i] > 0 ? 1.0 : (params[i] < 0 ? -1.0 : 0.0));
    }
  }
  
private:
  double lambda_;
};

class ElasticNetRegularizer : public Regularizer {
public:
  ElasticNetRegularizer(double l1_lambda = 0.01, double l2_lambda = 0.01) 
    : l1_(l1_lambda), l2_(l2_lambda) {}
  
  double compute_penalty(const Vector& params) const override {
    return l1_.compute_penalty(params) + l2_.compute_penalty(params);
  }
  
  void add_gradient(Vector& grad, const Vector& params) const override {
    l1_.add_gradient(grad, params);
    l2_.add_gradient(grad, params);
  }
  
private:
  L1Regularizer l1_;
  L2Regularizer l2_;
};

} // namespace regularizers

/**
 * @brief Advanced training objective with flexible components
 */
template<typename Model, typename ParamExtractor>
class AdvancedObjective {
public:
  using Dataset = std::vector<std::pair<Vector, Vector>>;
  
  AdvancedObjective(Model model, 
           std::unique_ptr<ParamExtractor> param_extractor,
           std::unique_ptr<LossFunction> loss_func,
           std::vector<std::unique_ptr<Regularizer>> regularizers = {})
    : model_(std::move(model)), 
      param_extractor_(std::move(param_extractor)),
      loss_func_(std::move(loss_func)),
      regularizers_(std::move(regularizers)),
      temp_output_(1) {}  // Will be resized as needed
  
  double compute_loss(const Vector& params, const Dataset& dataset) const {
    // Apply parameters to model
    param_extractor_->apply_parameters(model_, params);
    
    double total_loss = 0.0;
    
    // Compute data loss
    for (const auto& [input, target] : dataset) {
      // Forward pass
      forward_pass(input);
      
      // Add loss
      total_loss += loss_func_->compute_loss(temp_output_, target);
    }
    
    // Add regularization
    for (const auto& reg : regularizers_) {
      total_loss += reg->compute_penalty(params);
    }
    
    return total_loss / static_cast<double>(dataset.size());
  }
  
  void compute_gradient(Vector& grad, const Vector& params, const Dataset& dataset) const {
    // Apply parameters to model
    param_extractor_->apply_parameters(model_, params);
    
    grad.assign(params.size(), 0.0);
    Vector sample_grad(params.size());
    Vector output_grad;
    
    // Compute data gradient
    for (const auto& [input, target] : dataset) {
      // Forward pass
      forward_pass(input);
      
      // Backward pass - compute loss gradient w.r.t. output
      loss_func_->compute_gradient(output_grad, temp_output_, target);
      
      // Extract parameter gradient
      param_extractor_->extract_gradient(sample_grad, model_, output_grad);
      add_in_place(grad, sample_grad);
    }
    
    // Normalize by dataset size
    scale_in_place(grad, 1.0 / static_cast<double>(dataset.size()));
    
    // Add regularization gradients
    for (const auto& reg : regularizers_) {
      reg->add_gradient(grad, params);
    }
  }
  
  double value(const Vector& params) const {
    // For single-sample interface (backward compatibility)
    return 0.0; // Would need to store dataset
  }
  
  void gradient(Vector& g, const Vector& params) const {
    // For single-sample interface (backward compatibility)
    g.assign(params.size(), 0.0);
  }
  
private:
  void forward_pass(const Vector& input) const {
    // Get bias from parameter extractor if it's a BiasParameterExtractor
    if (auto bias_extractor = dynamic_cast<const BiasParameterExtractor<Model>*>(param_extractor_.get())) {
      temp_output_.resize(input.size());
      model_(temp_output_, input);
      
      Vector bias = bias_extractor->get_bias();
      if (!bias.empty()) {
        add_in_place(temp_output_, bias);
      }
    } else {
      temp_output_.resize(input.size());
      model_(temp_output_, input);
    }
  }
  
  mutable Model model_;
  std::unique_ptr<ParamExtractor> param_extractor_;
  std::unique_ptr<LossFunction> loss_func_;
  std::vector<std::unique_ptr<Regularizer>> regularizers_;
  mutable Vector temp_output_;
};

/**
 * @brief Factory functions for creating objectives
 */

// Simple MSE objective with bias parameters (backward compatible)
template<self_map_c<Vector> Model>
auto make_mse_objective(Model model, std::size_t output_dim) {
  return AdvancedObjective<Model, BiasParameterExtractor<Model>>(
    std::move(model),
    std::make_unique<BiasParameterExtractor<Model>>(output_dim),
    std::make_unique<loss_functions::MeanSquaredError>()
  );
}

// Cross-entropy objective with L2 regularization
template<self_map_c<Vector> Model>
auto make_crossentropy_objective(Model model, std::size_t output_dim, double l2_lambda = 0.01) {
  std::vector<std::unique_ptr<Regularizer>> regs;
  regs.push_back(std::make_unique<regularizers::L2Regularizer>(l2_lambda));
  
  return AdvancedObjective<Model, BiasParameterExtractor<Model>>(
    std::move(model),
    std::make_unique<BiasParameterExtractor<Model>>(output_dim),
    std::make_unique<loss_functions::CrossEntropy>(),
    std::move(regs)
  );
}

// Robust regression with Huber loss and Elastic Net regularization
template<self_map_c<Vector> Model>
auto make_robust_objective(Model model, std::size_t output_dim, 
              double huber_delta = 1.0, 
              double l1_lambda = 0.01, 
              double l2_lambda = 0.01) {
  std::vector<std::unique_ptr<Regularizer>> regs;
  regs.push_back(std::make_unique<regularizers::ElasticNetRegularizer>(l1_lambda, l2_lambda));
  
  return AdvancedObjective<Model, BiasParameterExtractor<Model>>(
    std::move(model),
    std::make_unique<BiasParameterExtractor<Model>>(output_dim),
    std::make_unique<loss_functions::HuberLoss>(huber_delta),
    std::move(regs)
  );
}

/**
 * @brief Advanced training with batch processing and multiple optimizers
 */
template<typename Objective>
class AdvancedTrainer {
public:
  using Dataset = typename Objective::Dataset;
  
  struct TrainingConfig {
    std::size_t max_iterations = 1000;
    double learning_rate = 1e-3;
    std::size_t batch_size = 32;
    double tolerance = 1e-6;
    std::size_t patience = 10;  // Early stopping
    bool verbose = false;
  };
  
  AdvancedTrainer(Objective objective, TrainingConfig config = {})
    : objective_(std::move(objective)), config_(config) {}
  
  Vector train(const Dataset& dataset, Vector initial_params = {}) {
    if (initial_params.empty()) {
      // Initialize parameters
      initial_params.resize(100, 0.0);  // Default size - should be configurable
      
      // Xavier initialization
      std::random_device rd;
      std::mt19937 gen(rd());
      double std_dev = std::sqrt(2.0 / initial_params.size());
      std::normal_distribution<> dist(0.0, std_dev);
      auto randomize = [&](auto) { return dist(gen); };
      unary_in_place(initial_params, randomize);
    }
    
    Vector params = initial_params;
    Vector gradient(params.size());
    Vector momentum(params.size(), 0.0);
    
    double best_loss = std::numeric_limits<double>::infinity();
    std::size_t patience_counter = 0;
    const double momentum_factor = 0.9;
    
    for (std::size_t iter = 0; iter < config_.max_iterations; ++iter) {
      // Create mini-batches
      auto batches = create_batches(dataset, config_.batch_size);
      
      double epoch_loss = 0.0;
      for (const auto& batch : batches) {
        // Compute gradient on batch
        objective_.compute_gradient(gradient, params, batch);
        
        // Update with momentum
        scale_in_place(momentum, momentum_factor);
        Vector lr_grad = rvf::clone(gradient);
        scale_in_place(lr_grad, -config_.learning_rate);
        add_in_place(momentum, lr_grad);
        add_in_place(params, momentum);
        
        epoch_loss += objective_.compute_loss(params, batch);
      }
      
      epoch_loss /= batches.size();
      
      if (config_.verbose && iter % 10 == 0) {
        std::cout << "Iteration " << iter << ", Loss: " << epoch_loss << std::endl;
      }
      
      // Early stopping
      if (epoch_loss < best_loss - config_.tolerance) {
        best_loss = epoch_loss;
        patience_counter = 0;
      } else {
        patience_counter++;
        if (patience_counter >= config_.patience) {
          if (config_.verbose) {
            std::cout << "Early stopping at iteration " << iter << std::endl;
          }
          break;
        }
      }
    }
    
    return params;
  }
  
private:
  std::vector<Dataset> create_batches(const Dataset& dataset, std::size_t batch_size) const {
    std::vector<Dataset> batches;
    
    for (std::size_t i = 0; i < dataset.size(); i += batch_size) {
      Dataset batch;
      std::size_t end = std::min(i + batch_size, dataset.size());
      batch.assign(dataset.begin() + i, dataset.begin() + end);
      batches.push_back(std::move(batch));
    }
    
    return batches;
  }
  
  Objective objective_;
  TrainingConfig config_;
};

/**
 * @brief Simplified training interface (backward compatible)
 */
template<self_map_c<Vector> Model, typename Dataset>
void train_with_rvf_advanced(Model& model, const Dataset& dataset, 
               std::size_t iterations = 200, double lr = 1e-3) {
  if (dataset.empty()) return;
  
  const std::size_t output_dim = dataset.front().second.size();
  auto objective = make_mse_objective(model, output_dim);
  
  typename AdvancedTrainer<decltype(objective)>::TrainingConfig config;
  config.max_iterations = iterations;
  config.learning_rate = lr;
  config.batch_size = std::min(static_cast<std::size_t>(32), dataset.size());
  config.verbose = true;
  
  AdvancedTrainer trainer(std::move(objective), config);
  Vector params = trainer.train(dataset);
  
  // The trained parameters are now available, but since the original interface
  // doesn't return them, we'd need to modify the model in-place or change the interface
}

// ============================================================================
// EXAMPLES OF SIMPLIFIED USAGE
// ============================================================================

/**
 * @brief Create various architectures using the simplified approach
 */
void demonstrate_simplicity() {
  const std::size_t d_model = 128;
  
  // 1. Simple feedforward network
  auto ffn = make_feed_forward(d_model, 512);
  
  // 2. Residual network  
  auto resnet_block = add_residual(
  add_layer_norm(
    compose(
    make_linear_transform(Matrix(d_model, Vector(d_model)), Vector(d_model)),
    make_activation(relu)
    )
  )
  );
  
  // 3. Multi-layer perceptron
  auto mlp = make_sequential(
  make_linear_transform(Matrix(512, Vector(d_model)), Vector(512)),
  make_activation(relu),
  make_linear_transform(Matrix(256, Vector(512)), Vector(256)), 
  make_activation(relu),
  make_linear_transform(Matrix(10, Vector(256)), Vector(10)),
  make_activation(softmax)  // Using RVF's softmax CPO!
  );
  
  // 4. Ensemble model
  auto ensemble = make_parallel_sum(
  make_feed_forward(d_model, 256),
  make_feed_forward(d_model, 512), 
  make_feed_forward(d_model, 1024)
  );
  
  // 5. Complete transformer
  auto transformer = make_transformer(d_model, 6, 8, 2048);
  
  // All of these satisfy self_map_c and can be used with RVF algorithms!
  static_assert(self_map_c<decltype(ffn), Vector>);
  static_assert(self_map_c<decltype(resnet_block), Vector>);
  static_assert(self_map_c<decltype(mlp), Vector>);
  static_assert(self_map_c<decltype(ensemble), Vector>);
  static_assert(self_map_c<decltype(transformer), Vector>);
}

} // namespace transformer
