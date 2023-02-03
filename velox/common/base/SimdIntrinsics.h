#pragma once

#include <cstdint>

#include <arm_neon.h>

// We still need this for convenience in this file. None of it is exposed, as we
// put it in the nested namespace.
namespace real_xsimd {
// #include <xsimd/xsimd.hpp>
// using namespace xsimd;
}

#define XSIMD_TEMPLATE template <typename T, typename A = default_arch>

// #define XSIMD_WITH_NEON 1

namespace xsimd {

struct default_arch;

struct generic {
  constexpr generic() = default;
  generic(default_arch arch);
};

struct default_arch : public generic {
  constexpr default_arch() = default;
  // TODO: 16 fails here:
  // /Users/law/repos/velox-lawben/./velox/common/base/SimdUtil-inl.h:222:13:
  static constexpr size_t alignment() noexcept {
    return 16;
  }
};

///////////////////////////////
///          TYPES          ///
///////////////////////////////

namespace types {
template <class T, class A>
struct has_simd_register : std::false_type {};

template <class T, class Arch>
struct simd_register {
  struct register_type {};
};

//    using vector_type __attribute__((vector_size(16))) = SCALAR_TYPE;

#define XSIMD_DECLARE_SIMD_REGISTER(SCALAR_TYPE, ISA) \
  template <>                                         \
  struct simd_register<SCALAR_TYPE, ISA> {            \
    using vector_type = uint8x16_t;                   \
    using register_type = vector_type;                \
    register_type data;                               \
    operator register_type() const noexcept {         \
      return data;                                    \
    }                                                 \
  };                                                  \
  template <>                                         \
  struct has_simd_register<SCALAR_TYPE, ISA> : std::true_type {}

#define XSIMD_DECLARE_INVALID_SIMD_REGISTER(SCALAR_TYPE, ISA) \
  template <>                                                 \
  struct has_simd_register<SCALAR_TYPE, ISA> : std::false_type {}

#define XSIMD_DECLARE_SIMD_REGISTER_ALIAS(ISA, ISA_BASE)                      \
  template <class T>                                                          \
  struct simd_register<T, ISA> : simd_register<T, ISA_BASE> {                 \
    using register_type = typename simd_register<T, ISA_BASE>::register_type; \
    simd_register(register_type reg) noexcept                                 \
        : simd_register<T, ISA_BASE>{reg} {}                                  \
    simd_register() = default;                                                \
  };                                                                          \
  template <class T>                                                          \
  struct has_simd_register<T, ISA> : has_simd_register<T, ISA_BASE> {}

XSIMD_DECLARE_SIMD_REGISTER(signed char, generic);
XSIMD_DECLARE_SIMD_REGISTER(unsigned char, generic);
XSIMD_DECLARE_SIMD_REGISTER(char, generic);
XSIMD_DECLARE_SIMD_REGISTER(short, generic);
XSIMD_DECLARE_SIMD_REGISTER(unsigned short, generic);
XSIMD_DECLARE_SIMD_REGISTER(int, generic);
XSIMD_DECLARE_SIMD_REGISTER(unsigned int, generic);
XSIMD_DECLARE_SIMD_REGISTER(long int, generic);
XSIMD_DECLARE_SIMD_REGISTER(unsigned long int, generic);
XSIMD_DECLARE_SIMD_REGISTER(long long int, generic);
XSIMD_DECLARE_SIMD_REGISTER(unsigned long long int, generic);
XSIMD_DECLARE_SIMD_REGISTER(float, generic);
XSIMD_DECLARE_SIMD_REGISTER(bool, generic);

template <class T, class Arch>
struct get_bool_simd_register {
  using type = simd_register<T, Arch>;
};

template <class T, class Arch>
using get_bool_simd_register_t = typename get_bool_simd_register<T, Arch>::type;

namespace detail {
template <size_t S>
struct get_unsigned_type;

template <>
struct get_unsigned_type<1> {
  using type = uint8_t;
};

template <>
struct get_unsigned_type<2> {
  using type = uint16_t;
};

template <>
struct get_unsigned_type<4> {
  using type = uint32_t;
};

template <>
struct get_unsigned_type<8> {
  using type = uint64_t;
};

template <size_t S>
using get_unsigned_type_t = typename get_unsigned_type<S>::type;

template <class T, class A>
struct neon_bool_simd_register {
  using type = simd_register<get_unsigned_type_t<sizeof(T)>, A>;
};
} // namespace detail

template <class T>
struct get_bool_simd_register<T, generic>
    : detail::neon_bool_simd_register<T, generic> {};

} // namespace types

XSIMD_TEMPLATE
struct has_simd_register : types::has_simd_register<T, A> {};

struct avx {};
struct avx2 {};
struct sse2 {};
struct neon {};

XSIMD_TEMPLATE
struct batch;

XSIMD_TEMPLATE
struct batch_bool : public types::get_bool_simd_register_t<T, A> {
  static constexpr size_t size = 16 / sizeof(T);

  using base_type = types::get_bool_simd_register_t<T, A>;
  using value_type = bool;
  using arch_type = A;
  using register_type = typename base_type::register_type;
  using batch_type = batch<T, A>;

  batch_bool() = default;
  batch_bool(bool val) noexcept;
  batch_bool(register_type reg) noexcept;
  batch_bool(batch_type batch) noexcept;
  template <class... Ts>
  batch_bool(bool val0, bool val1, Ts... vals) noexcept;

  // comparison operators
  batch_bool operator==(const batch_bool& other) const noexcept;
  batch_bool operator!=(const batch_bool& other) const noexcept;

  // logical operators
  batch_bool operator~() const noexcept;
  batch_bool operator!() const noexcept;
  batch_bool operator&(const batch_bool& other) const noexcept;
  batch_bool operator|(const batch_bool& other) const noexcept;
  batch_bool operator^(const batch_bool& other) const noexcept;
  batch_bool operator&&(const batch_bool& other) const noexcept;
  batch_bool operator||(const batch_bool& other) const noexcept;

  // update operators
  batch_bool& operator&=(const batch_bool& other) const noexcept {
    return (*this) = (*this) & other;
  }
  batch_bool& operator|=(const batch_bool& other) const noexcept {
    return (*this) = (*this) | other;
  }
  batch_bool& operator^=(const batch_bool& other) const noexcept {
    return (*this) = (*this) ^ other;
  }

  void store_aligned(void* dst);
  void store_unaligned(void* dst);

  static batch_bool load_aligned(void* dst);
  static batch_bool load_unaligned(void* dst);
};

template <typename T, typename A>
struct batch : public types::simd_register<T, A> {
  static constexpr size_t size = 16 / sizeof(T);
  using arch_type = A;
  using batch_bool_type = batch_bool<T, A>;
  using register_type = typename types::simd_register<T, A>::register_type;

  batch() = default; ///< Create a batch initialized with undefined values.
  batch(T val) noexcept;

  explicit batch(batch_bool_type const& b) noexcept;
  batch(register_type reg) noexcept;

  template <class... Ts>
  batch(T val0, T val1, Ts... vals) noexcept;

  batch_bool<T, A> operator==(const batch& other) const noexcept;
  batch_bool<T, A> operator!=(const batch& other) const noexcept;
  batch_bool<T, A> operator>=(const batch& other) const noexcept;
  batch_bool<T, A> operator<=(const batch& other) const noexcept;
  batch_bool<T, A> operator>(const batch& other) const noexcept;
  batch_bool<T, A> operator<(const batch& other) const noexcept;

  batch operator^(const batch& other) const noexcept;
  batch operator&(const batch& other) const noexcept;
  batch operator*(const batch& other) const noexcept;
  batch operator+(const batch& other) const noexcept;
  batch operator-(const batch& other) const noexcept;
  batch operator<<(const batch& other) const noexcept;
  batch operator>>(const batch& other) const noexcept;

  static batch broadcast(T value);

  template <typename U>
  void store_aligned(U* dst);

  template <typename U>
  void store_unaligned(U* dst);

  template <typename U>
  static batch load_aligned(const U* dst);

  template <typename U>
  static batch load_unaligned(const U* dst);
};

///////////////////////////////
///         METHODS         ///
///////////////////////////////
XSIMD_TEMPLATE
batch<T, A> broadcast(T value);

// XSIMD_TEMPLATE
// batch<T, A> load_aligned(const void* addr);
//
// XSIMD_TEMPLATE
// batch<T, A> load_unaligned(const void* addr);

template <class A = default_arch, class From>
batch<From, A> load_aligned(const From* ptr) noexcept;

template <class A = default_arch, class From>
batch<From, A> load_unaligned(const From* ptr) noexcept;

template <typename T, std::size_t N>
using make_sized_batch_t = batch<T, default_arch>;

} // namespace xsimd
