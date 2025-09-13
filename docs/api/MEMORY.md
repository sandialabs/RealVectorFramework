# RVF Memory Management Reference

This document provides comprehensive reference for RVF's advanced memory management systems, including memory arenas, allocation strategies, and performance optimization techniques.

## Overview

RVF provides sophisticated memory management capabilities designed for high-performance numerical computing scenarios where frequent vector allocations can become a bottleneck. The memory management system includes:

- Memory arenas for pool-based allocation
- Arena integration with CPOs
- Performance observers and monitoring
- Automatic lifetime management

---

## Memory Arena System

### `MemoryArena` Class

```cpp
class MemoryArena {
public:
    explicit MemoryArena(size_t capacity_bytes);
    ~MemoryArena();

    // Scoped arena management
    auto create_scope() -> ScopedArena;

    // Memory allocation interface
    void* allocate(size_t bytes, size_t alignment = alignof(std::max_align_t));
    void deallocate(void* ptr, size_t bytes) noexcept;

    // Statistics and monitoring
    auto get_statistics() const -> ArenaStatistics;
    void reset();

    // Configuration
    void set_growth_policy(GrowthPolicy policy);
    void set_alignment_policy(AlignmentPolicy policy);

private:
    // Implementation details hidden
};
```

**Purpose**: Provides fast memory allocation from pre-allocated pools.

**Key Features**:
- **Pool-based allocation**: Fast O(1) allocation for many scenarios
- **Automatic growth**: Pools expand as needed
- **Scope management**: RAII-based automatic cleanup
- **Statistics tracking**: Monitor memory usage patterns
- **Thread safety**: Optional thread-local or synchronized pools

### Constructor

```cpp
explicit MemoryArena(size_t capacity_bytes);
```

**Parameters**:
- `capacity_bytes`: Initial pool capacity in bytes

**Example**:
```cpp
// Create 1MB arena
rvf::MemoryArena arena(1024 * 1024);

// Create larger arena for big problems
rvf::MemoryArena big_arena(128 * 1024 * 1024);  // 128MB
```

### Scoped Arena Management

#### `ScopedArena` Class

```cpp
class ScopedArena {
public:
    ScopedArena(ScopedArena&& other) noexcept;
    ~ScopedArena();  // Automatic cleanup

    // Disable copy
    ScopedArena(const ScopedArena&) = delete;
    ScopedArena& operator=(const ScopedArena&) = delete;

    // Statistics access
    auto get_scope_statistics() const -> ScopeStatistics;
};
```

**Purpose**: RAII wrapper for arena scopes with automatic cleanup.

**Usage Pattern**:
```cpp
rvf::MemoryArena arena(1024 * 1024);

{
    auto scoped_arena = arena.create_scope();

    // All arena allocations within this scope
    auto v1 = arena_clone(vector1);  // Arena allocation
    auto v2 = arena_clone(vector2);  // Arena allocation

    // ... perform computations ...

} // Automatic cleanup - all scope allocations released
```

### Arena Statistics

#### `ArenaStatistics` Structure

```cpp
struct ArenaStatistics {
    size_t total_capacity;        // Total arena capacity
    size_t bytes_allocated;       // Currently allocated bytes
    size_t bytes_available;       // Available bytes
    size_t peak_usage;           // Peak memory usage
    size_t total_allocations;    // Total allocation count
    size_t failed_allocations;   // Failed allocation count
    double fragmentation_ratio;  // Fragmentation metric
    std::chrono::microseconds avg_allocation_time;
};
```

**Usage**:
```cpp
auto stats = arena.get_statistics();
std::cout << "Memory usage: " << stats.bytes_allocated << "/"
          << stats.total_capacity << " bytes\n";
std::cout << "Fragmentation: " << stats.fragmentation_ratio << "\n";
```

#### `ScopeStatistics` Structure

```cpp
struct ScopeStatistics {
    size_t scope_allocations;     // Allocations in current scope
    size_t scope_bytes;          // Bytes allocated in scope
    std::chrono::microseconds scope_lifetime;
};
```

---

## Arena Integration CPOs

### `arena_integration`

```cpp
TINCUP_DEFINE_CPO(arena_integration_ftor, arena_integration)
```

**Purpose**: Provides CPO hooks for arena-aware vector operations.

**Specialization Interface**:
```cpp
// Custom vector with arena support
struct ArenaVector {
    static ArenaVector allocate_from_arena(MemoryArena& arena, size_t size);
    void deallocate_to_arena(MemoryArena& arena);
};

// CPO specialization
auto tag_invoke(rvf::arena_integration_ftor, rvf::clone_ftor,
                const ArenaVector& v, MemoryArena& arena) -> ArenaVector {
    return ArenaVector::allocate_from_arena(arena, v.size());
}
```

### `arena_observers`

```cpp
TINCUP_DEFINE_CPO(arena_observers_ftor, arena_observers)
```

**Purpose**: Provides hooks for monitoring and observing arena operations.

**Observer Interface**:
```cpp
struct ArenaObserver {
    void on_allocation(void* ptr, size_t bytes, size_t alignment);
    void on_deallocation(void* ptr, size_t bytes);
    void on_scope_enter(const ScopedArena& scope);
    void on_scope_exit(const ScopedArena& scope);
};
```

**Registration**:
```cpp
// Register observer
arena_observers(arena, my_observer);

// Builtin observers
auto perf_observer = make_performance_observer();
arena_observers(arena, perf_observer);
```

---

## CPO Arena Integration

### Arena-Aware Clone Operation

```cpp
template<real_vector_c Vector>
auto arena_clone(const Vector& v, MemoryArena& arena) -> /* arena-allocated vector */;
```

**Purpose**: Clone vector using arena allocation.

**Implementation Strategy**:
1. Check for CPO specialization with arena support
2. Fall back to regular clone if no arena support
3. Automatic integration with scoped arenas

**Example**:
```cpp
rvf::MemoryArena arena(1024 * 1024);

{
    auto scope = arena.create_scope();

    std::vector<double> original{1, 2, 3, 4, 5};

    // Arena-allocated clone
    auto arena_copy = arena_clone(original, arena);
    auto& copy_ref = deref_if_needed(arena_copy);

    // Use copy_ref normally
    scale_in_place(copy_ref, 2.0);

} // arena_copy automatically deallocated
```

### Thread-Local Arena Support

```cpp
class ThreadLocalArena {
public:
    static auto get_thread_arena() -> MemoryArena&;
    static void set_thread_arena_capacity(size_t bytes);

    // Convenience functions
    template<real_vector_c Vector>
    static auto clone(const Vector& v) -> /* arena-allocated vector */;
};
```

**Usage**:
```cpp
// Set thread-local arena size
ThreadLocalArena::set_thread_arena_capacity(64 * 1024 * 1024);

// Use thread-local arena automatically
void worker_thread() {
    std::vector<double> data(10000);

    for (int i = 0; i < 1000; ++i) {
        auto temp = ThreadLocalArena::clone(data);  // Fast allocation
        // ... computation ...
    }
    // All temp vectors automatically cleaned up
}
```

---

## Performance Optimization

### Allocation Strategies

#### Stack-Based Allocation

```cpp
template<typename T, size_t N>
class StackVector {
    alignas(T) char storage[N * sizeof(T)];
    size_t size_;

public:
    // Optimized for small, fixed-size vectors
    static constexpr size_t max_size = N;

    // Zero allocation cost
    StackVector(size_t size) : size_(size) {
        assert(size <= N);
    }
};
```

**Use Cases**:
- Small vectors (< 1KB)
- Inner loop temporaries
- Known maximum sizes

#### Memory Pool Strategies

```cpp
enum class GrowthPolicy {
    FIXED_SIZE,      // No growth, fail when full
    LINEAR_GROWTH,   // Grow by fixed amount
    EXPONENTIAL,     // Double size when full
    ADAPTIVE         // Heuristic-based growth
};

enum class AlignmentPolicy {
    NATURAL,         // Natural alignment for type
    CACHE_LINE,      // Align to cache line boundaries
    PAGE_ALIGNED,    // Align to page boundaries
    CUSTOM           // User-specified alignment
};
```

### Performance Benchmarks

Based on included benchmark results:

#### Arena vs Standard Allocation

| Vector Size | Standard Alloc | Arena Alloc | Speedup |
|-------------|----------------|-------------|---------|
| 100 elements | 35.4 ns | 65.1 ns | 0.54x |
| 10K elements | 1993 ns | 65.7 ns | **30.3x** |
| 1M elements | 358504 ns | 65.0 ns | **5516x** |

#### Batch Operations

| Operation | Standard | Arena | Speedup |
|-----------|----------|-------|---------|
| 1000 elem, 10 batch | 1885 ns | 682 ns | **2.76x** |
| 1000 elem, 100 batch | 19185 ns | 6493 ns | **2.95x** |
| 10000 elem, 100 batch | 347120 ns | 6554 ns | **52.9x** |

### Memory Usage Patterns

#### Algorithm Pattern Analysis

```cpp
// Typical iterative algorithm pattern
template<real_vector_c Vector>
void iterative_algorithm(Vector& x, int max_iter) {
    MemoryArena arena(estimate_memory_needs(x, max_iter));

    for (int iter = 0; iter < max_iter; ++iter) {
        auto scope = arena.create_scope();

        auto temp1 = arena_clone(x, arena);  // Arena allocation
        auto& t1 = deref_if_needed(temp1);

        auto temp2 = arena_clone(x, arena);  // Arena allocation
        auto& t2 = deref_if_needed(temp2);

        // Perform iteration operations
        scale_in_place(t1, 2.0);
        axpy_in_place(t2, 1.5, t1);
        add_in_place(x, t2);

    } // All temporaries automatically deallocated
}
```

**Benefits**:
- **10-100x faster** allocation for medium/large vectors
- **Reduced fragmentation** in long-running algorithms
- **Better cache locality** due to memory pool organization
- **Predictable performance** - no malloc/free overhead

---

## Integration Examples

### Custom Vector Type with Arena Support

```cpp
template<typename T>
class ArenaAwareVector {
    T* data_;
    size_t size_;
    MemoryArena* arena_;  // nullptr for non-arena allocation

public:
    // Regular constructor
    ArenaAwareVector(size_t size)
        : size_(size), arena_(nullptr) {
        data_ = new T[size];
    }

    // Arena constructor
    ArenaAwareVector(size_t size, MemoryArena& arena)
        : size_(size), arena_(&arena) {
        data_ = static_cast<T*>(arena.allocate(size * sizeof(T), alignof(T)));
    }

    ~ArenaAwareVector() {
        if (arena_) {
            arena_->deallocate(data_, size_ * sizeof(T));
        } else {
            delete[] data_;
        }
    }

    // Vector interface
    size_t size() const { return size_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
};

// CPO specializations
auto tag_invoke(rvf::clone_ftor, const ArenaAwareVector<double>& v)
    -> ArenaAwareVector<double> {
    return ArenaAwareVector<double>(v.size());
}

auto tag_invoke(rvf::arena_integration_ftor, rvf::clone_ftor,
                const ArenaAwareVector<double>& v, MemoryArena& arena)
    -> ArenaAwareVector<double> {
    return ArenaAwareVector<double>(v.size(), arena);
}
```

### High-Performance Computing Workflow

```cpp
void hpc_simulation() {
    const size_t n = 1000000;  // Large problem size
    const int time_steps = 10000;

    // Estimate memory needs
    size_t estimated_memory = n * sizeof(double) * 10;  // 10 temp vectors per step
    MemoryArena arena(estimated_memory);

    std::vector<double> state(n);
    initialize_state(state);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < time_steps; ++step) {
        auto scope = arena.create_scope();

        // All temporaries use arena
        auto temp1 = arena_clone(state, arena);
        auto temp2 = arena_clone(state, arena);
        auto temp3 = arena_clone(state, arena);

        // Dereference for operations
        auto& t1 = deref_if_needed(temp1);
        auto& t2 = deref_if_needed(temp2);
        auto& t3 = deref_if_needed(temp3);

        // Physics simulation step
        compute_derivatives(t1, state);
        compute_forces(t2, state);
        time_integration_step(state, t1, t2, t3);

        // Scope cleanup is automatic
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    auto stats = arena.get_statistics();
    std::cout << "Simulation completed in " << duration.count() << " ms\n";
    std::cout << "Peak memory usage: " << stats.peak_usage << " bytes\n";
    std::cout << "Total allocations: " << stats.total_allocations << "\n";
}
```

---

## Best Practices

### Memory Arena Sizing

```cpp
template<real_vector_c Vector>
size_t estimate_arena_size(const Vector& template_vec, int algorithm_steps) {
    size_t element_size = sizeof(typename Vector::value_type);
    size_t vector_size = dimension(template_vec);
    size_t vectors_per_step = 5;  // Algorithm-specific

    return vector_size * element_size * vectors_per_step * algorithm_steps;
}
```

### Error Handling

```cpp
void safe_arena_usage() {
    MemoryArena arena(1024 * 1024);

    try {
        auto scope = arena.create_scope();

        // Arena operations may throw on allocation failure
        auto large_vector = arena_allocate<double>(arena, 1000000000);

    } catch (const std::bad_alloc& e) {
        std::cerr << "Arena allocation failed: " << e.what() << "\n";
        // Fallback to standard allocation
    }
}
```

### Performance Monitoring

```cpp
class PerformanceTracker {
    std::chrono::high_resolution_clock::time_point start_time;
    MemoryArena* arena_;

public:
    PerformanceTracker(MemoryArena& arena) : arena_(&arena) {
        start_time = std::chrono::high_resolution_clock::now();
    }

    ~PerformanceTracker() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);

        auto stats = arena_->get_statistics();
        std::cout << "Execution time: " << duration.count() << " μs\n";
        std::cout << "Memory efficiency: "
                  << (1.0 - stats.fragmentation_ratio) * 100 << "%\n";
    }
};

void monitored_computation() {
    MemoryArena arena(64 * 1024 * 1024);
    PerformanceTracker tracker(arena);

    // Computation using arena
    iterative_algorithm_with_arena(arena);
}
```