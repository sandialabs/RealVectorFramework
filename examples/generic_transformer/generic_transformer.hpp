#pragma once

#include "core/rvf.hpp"
#include "core/type_support/std_cpo_impl.hpp"
#include "operations/advanced/self_map.hpp"
#include "operations/advanced/relu.hpp"
#include "operations/advanced/softmax.hpp" 
#include "operations/advanced/layer_norm.hpp"
#include "operations/advanced/matvec.hpp"
#include "core/cmath/sqrt.hpp"
#include <vector>
#include <random>
#include <functional>

namespace transformer {

using namespace rvf;

template<typename Scalar, typename VectorType>
  requires rvf::real_scalar_c<Scalar> && rvf::real_vector_c<VectorType>
class Transformer {
public:
  using Vector = VectorType;
  using Matrix = std::vector<VectorType>;

// ============================================================================
// FUNCTIONAL COMPOSITION APPROACH - MUCH SIMPLER!
// ============================================================================

/**
 * @brief Generic linear transformation as a simple callable
 * 
 * This is much simpler than the previous class-based approach.
 * It's just a function that captures weights and uses RVF CPOs.
 */
auto make_linear_transform(const Matrix& weights, const Vector& bias) {
  return [weights, bias](Vector& y, const Vector& x) {
    matvec(y, weights, x);      // RVF's native matvec
    add_in_place(y, bias);      // RVF's native add_in_place
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
    y = x;              // Copy input
    activation_cpo(y);  // Apply activation using RVF CPO
  };
}

/**
 * @brief Compose two self_map_c operators sequentially
 * 
 * This is the key insight: we can compose any self_map_c operators!
 * This makes building complex architectures trivial.
 */
template<self_map_c<Vector> Op1, self_map_c<Vector> Op2>
auto compose(Op1&& op1, Op2&& op2) {
  return [op1 = std::forward<Op1>(op1), op2 = std::forward<Op2>(op2)]
       (Vector& y, const Vector& x) {
    Vector temp(x.size());
    op1(temp, x);      // Apply first operation
    op2(y, temp);      // Apply second operation  
  };
}

/**
 * @brief Add residual connection to any self_map_c operator
 * 
 * Residual(f)(x) = f(x) + x
 * This works with ANY self_map_c operator!
 */
template<self_map_c<Vector> Op>
auto add_residual(Op&& op) {
  return [op = std::forward<Op>(op)](Vector& y, const Vector& x) {
    op(y, x);              // Apply operation
    add_in_place(y, x);    // Add residual using RVF CPO
  };
}

/**
 * @brief Add layer normalization to any self_map_c operator
 * 
 * LayerNorm(f)(x) = layer_norm(f(x))
 * Again, works with ANY self_map_c operator!
 */
template<self_map_c<Vector> Op>
auto add_layer_norm(Op&& op, Scalar eps = 1e-5) {
  return [op = std::forward<Op>(op), eps](Vector& y, const Vector& x) {
    op(y, x);           // Apply operation
    layer_norm(y, eps); // Normalize using RVF CPO
  };
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
  Vector b1(d_ff, Scalar(0.0));
  Matrix W2(d_model, Vector(d_ff));  
  Vector b2(d_model, Scalar(0.0));
  
  // Xavier initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  auto init_matrix = [&](Matrix& W, std::size_t in_dim, std::size_t out_dim) {
    Scalar std_dev = rvf::sqrt(Scalar(2.0) / (in_dim + out_dim));
    std::normal_distribution<Scalar> dist(Scalar(0.0), std_dev);
    for (auto& row : W) {
      for (auto& w : row) {
        w = dist(gen);
      }
    }
  };
  
  init_matrix(W1, d_model, d_ff);
  init_matrix(W2, d_ff, d_model);
  
  // Compose operations using RVF CPOs:
  // FFN(x) = W2 * ReLU(W1 * x + b1) + b2
  return compose(
    make_linear_transform(W1, b1),   // First linear layer
    compose(
      make_activation(relu),         // ReLU activation (RVF CPO!)
      make_linear_transform(W2, b2)  // Second linear layer
    )
  );
}

/**
 * @brief Create simplified attention mechanism
 * 
 * This is a pedagogical simplification focusing on the RVF integration
 */
auto make_attention(std::size_t d_model, std::size_t n_heads) {
  std::size_t d_k = d_model / n_heads;
  
  // Initialize projection matrices
  Matrix W_q(d_model, Vector(d_model));
  Matrix W_k(d_model, Vector(d_model));  
  Matrix W_v(d_model, Vector(d_model));
  Matrix W_o(d_model, Vector(d_model));
  
  // Xavier initialization
  std::random_device rd;
  std::mt19937 gen(rd());
  auto init_matrix = [&](Matrix& W, std::size_t in_dim, std::size_t out_dim) {
    Scalar std_dev = rvf::sqrt(Scalar(2.0) / (in_dim + out_dim));
    std::normal_distribution<Scalar> dist(Scalar(0.0), std_dev);
    auto randomize = [&](auto){ return dist(gen); }; 
    for (auto& row : W) unary_in_place(row, randomize);
  };

  init_matrix(W_q, d_model, d_model);
  init_matrix(W_k, d_model, d_model);
  init_matrix(W_v, d_model, d_model);
  init_matrix(W_o, d_model, d_model);
  
  return [W_q, W_k, W_v, W_o, d_k](Vector& y, const Vector& x) {
    Vector Q(x.size()), K(x.size()), V(x.size());
    
    // Project to Q, K, V using RVF's matvec
    matvec(Q, W_q, x);
    matvec(K, W_k, x);  
    matvec(V, W_v, x);
    
    // Simplified attention computation
    Scalar attention_score = inner_product(Q, K) / rvf::sqrt(Scalar(d_k));
    Vector scores = {attention_score};
    softmax(scores);  // RVF's native softmax CPO!
    
    // Apply attention and output projection  
    auto weighted_V = rvf::clone(V); 
    scale_in_place(weighted_V, scores[0]);
    matvec(y, W_o, weighted_V);  // RVF's native matvec
  };
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
        add_layer_norm(add_residual(attention)),           // Self-attention + residual + layer norm
        add_layer_norm(add_residual(feed_forward))         // Feed-forward + residual + layer norm  
    );
}

/**
 * @brief Create complete transformer using functional composition
 * 
 * This demonstrates the power of the functional approach:
 * building a complex model by composing simple operations!
 */
auto make_transformer(std::size_t d_model, std::size_t n_layers, std::size_t n_heads, std::size_t d_ff) {
  // Start with identity transformation
  std::function<void(Vector&, const Vector&)> transformer = [](Vector& y, const Vector& x) { y = x; };
  
  // Compose multiple transformer blocks
  for (std::size_t i = 0; i < n_layers; ++i) {
    auto block = make_transformer_block(d_model, n_heads, d_ff);
    transformer = compose(std::move(transformer), std::move(block));
  }
  
  return transformer;
}

// ============================================================================
// ULTRA-GENERIC ARCHITECTURE BUILDER
// ============================================================================

/**
 * @brief Generic sequential architecture builder
 * 
 * This allows building ANY sequential architecture using RVF operations!
 * Much more general than transformer-specific code.
 */
template<self_map_c<Vector>... Ops>
auto make_sequential(Ops&&... ops) {
  return [ops...](Vector& y, const Vector& x) {
    Vector temp1 = x, temp2;
    Vector* current = &temp1;
    Vector* next = &temp2;
    
    // Apply operations sequentially
    ((ops(*next, *current), std::swap(current, next)), ...);
    
    y = *current;
  };
}

/**
 * @brief Generic parallel architecture builder  
 * 
 * This allows building ensemble models, multi-branch architectures, etc.
 */
template<self_map_c<Vector>... Ops>
auto make_parallel_sum(Ops&&... ops) {
    return [ops...](Vector& y, const Vector& x) {
        y.assign(x.size(), 0.0);  // Initialize to zero
        Vector temp(x.size());
        
        // Apply each operation and sum results
        ((ops(temp, x), add_in_place(y, temp)), ...);
    };
}

// ============================================================================
// TRAINING WITH GENERIC RVF OPTIMIZATION
// ============================================================================

/**
 * @brief Generic training objective for any self_map_c model
 * 
 * This works with ANY model that satisfies self_map_c!
 */
template<self_map_c<Vector> Model>
class GenericObjective {
private:
    Model model_;
    Vector input_;
    Vector target_;
    mutable Vector temp_;
    
public:
    GenericObjective(Model model, const Vector& input, const Vector& target)
        : model_(std::move(model)), input_(input), target_(target), temp_(input.size()) {}
    
    // Interpret params as an output bias of length equal to target size.
    // Objective: 0.5 * || (model(input) + bias) - target ||^2
    Scalar value(const Vector& params) const {
        model_(temp_, input_);

        // y = model(input) + bias(params[0:target_.size()])
        Vector y = rvf::clone(temp_);
        const std::size_t m = target_.size();
        Vector bias;
        bias.assign(params.begin(), params.begin() + std::min(m, params.size()));
        add_in_place(y, bias);

        // diff = y - target
        Vector diff = rvf::clone(y);
        Vector neg_t(target_.begin(), target_.end());
        scale_in_place(neg_t, Scalar(-1.0));
        add_in_place(diff, neg_t);

        return Scalar(0.5) * inner_product(diff, diff);
    }

    // Gradient w.r.t. params (bias): dL/db = (model(input) + b) - target
    void gradient(Vector& g, const Vector& params) const {
        model_(temp_, input_);

        // y = model(input) + bias
        Vector y = rvf::clone(temp_);
        const std::size_t m = target_.size();
        Vector bias;
        bias.assign(params.begin(), params.begin() + std::min(m, params.size()));
        add_in_place(y, bias);

        // g = 0; g[0:m] = y - target
        if (g.size() < params.size()) g.resize(params.size());
        fill(g, Scalar(0.0));
        Vector diff = rvf::clone(y);
        Vector neg_t(target_.begin(), target_.end());
        scale_in_place(neg_t, Scalar(-1.0));
        add_in_place(diff, neg_t);
        for (std::size_t i = 0; i < std::min(m, g.size()); ++i) g[i] = diff[i];
    }
};

/**
 * @brief Train any self_map_c model with RVF optimization
 */
template<self_map_c<Vector> Model, typename Dataset>
void train_with_rvf(Model& model, const Dataset& dataset, std::size_t iterations = 1000) {
    // Size bias parameters to the target dimension of the first sample
    std::size_t param_dim = 0;
    if (!dataset.empty()) {
        const auto& first = dataset.front();
        param_dim = first.second.size();
    }
    if (param_dim == 0) return;

    Vector params(param_dim, Scalar(0.0));  // Initialize bias to zeros
    Vector gradient(param_dim, Scalar(0.0));
    
    for (std::size_t iter = 0; iter < iterations; ++iter) {
        // Compute gradients using RVF operations
        fill(gradient, Scalar(0.0));
        Scalar loss = Scalar(0.0);
        
        for (const auto& [input, target] : dataset) {
            GenericObjective objective(model, input, target);
            Vector sample_grad(param_dim, Scalar(0.0));
            loss += objective.value(params);
            objective.gradient(sample_grad, params);
            add_in_place(gradient, sample_grad);
        }
        
        // Simple SGD update
        scale_in_place(gradient, Scalar(-0.001));  // Learning rate
        add_in_place(params, gradient);    // Parameter update
        
        // Could also use RVF's trust region methods, conjugate gradient, etc.!
    }
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
          make_linear_transform(Matrix(d_model, Vector(d_model)), Vector(d_model, Scalar(0.0))),
          make_activation(relu)
        )
      )
    );
    
    // 3. Multi-layer perceptron
    auto mlp = make_sequential(
      make_linear_transform(Matrix(512, Vector(d_model)), Vector(512, Scalar(0.0))),
      make_activation(relu),
      make_linear_transform(Matrix(256, Vector(512)), Vector(256, Scalar(0.0))), 
      make_activation(relu),
      make_linear_transform(Matrix(10, Vector(256)), Vector(10, Scalar(0.0))),
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
}

} // namespace transformer
