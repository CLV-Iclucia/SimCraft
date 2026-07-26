#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ksk::core {

class dynamic_bitset {
 public:
  dynamic_bitset() = default;
  explicit dynamic_bitset(std::size_t bit_count)
      : bit_count_(bit_count),
        words_(wordCount(bit_count), std::uint64_t{0})
  {
  }

  [[nodiscard]] std::size_t size() const noexcept { return bit_count_; }
  [[nodiscard]] bool empty() const noexcept { return bit_count_ == 0; }

  void resize(std::size_t bit_count, bool value = false)
  {
    bit_count_ = bit_count;
    words_.assign(wordCount(bit_count),
                  value ? ~std::uint64_t{0} : std::uint64_t{0});
    trimUnusedBits();
  }

  void reset() noexcept
  {
    for (std::uint64_t& word : words_) {
      word = 0;
    }
  }

  void set(std::size_t bit, bool value = true)
  {
    check(bit);
    const std::uint64_t mask = bitMask(bit);
    std::uint64_t& word = words_[wordIndex(bit)];
    if (value) {
      word |= mask;
    } else {
      word &= ~mask;
    }
  }

  [[nodiscard]] bool test(std::size_t bit) const
  {
    check(bit);
    return (words_[wordIndex(bit)] & bitMask(bit)) != 0;
  }

 private:
  static constexpr int kWordBits = 64;

  [[nodiscard]] static std::size_t wordCount(std::size_t bit_count) noexcept
  {
    return (bit_count + kWordBits - 1) / kWordBits;
  }

  [[nodiscard]] static std::size_t wordIndex(std::size_t bit) noexcept
  {
    return bit / kWordBits;
  }

  [[nodiscard]] static std::uint64_t bitMask(std::size_t bit) noexcept
  {
    return std::uint64_t{1} << (bit % kWordBits);
  }

  void check(std::size_t bit) const
  {
    if (bit >= bit_count_) {
      throw std::out_of_range("DynamicBitset bit index is out of range");
    }
  }

  void trimUnusedBits() noexcept
  {
    if (words_.empty() || bit_count_ % kWordBits == 0) {
      return;
    }
    const std::uint64_t used_bits = bit_count_ % kWordBits;
    words_.back() &= (std::uint64_t{1} << used_bits) - 1;
  }

  std::size_t bit_count_ = 0;
  std::vector<std::uint64_t> words_;
};

}  // namespace ksk::core
