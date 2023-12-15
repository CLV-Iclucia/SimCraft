//
// Created by creeper on 10/24/23.
//

#ifndef SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_ARRAYS_H_
#define SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_ARRAYS_H_

#include <vector>
namespace spatify {
enum class PaddingMode {
  Around,
};

template<typename T, int Padding, PaddingMode = PaddingMode::Around>
class GhostArray2D {
 public:
  GhostArray2D() = default;
  GhostArray2D(int width, int height) : m_width(width), m_height(height) {
    m_data.resize((width + 2 * Padding) * (height + 2 * Padding));
  }
  T &operator()(int i, int j) {
    return m_data[(i + Padding) + (j + Padding) * (m_width + 2 * Padding)];
  }
  const T &operator()(int i, int j) const {
    return m_data[(i + Padding) + (j + Padding) * (m_width + 2 * Padding)];
  }
  T &operator()(const Vec2i &p) {
    return m_data[(p.x + Padding) + (p.y + Padding) * (m_width + 2 * Padding)];
  }
  const T &operator()(const Vec2i &p) const {
    return m_data[(p.x + Padding) + (p.y + Padding) * (m_width + 2 * Padding)];
  }
  T &operator[](int i) { return m_data[i]; }
  const T &operator[](int i) const { return m_data[i]; }
  int width() const { return m_width; }
  int height() const { return m_height; }
  int size() const { return m_data.size(); }
  void resize(int width, int height) {
    m_width = width;
    m_height = height;
    m_data.resize((width + 2 * Padding) * (height + 2 * Padding));
  }
  void resize(int width, int height, const T &value) {
    m_width = width;
    m_height = height;
    m_data.resize((width + 2 * Padding) * (height + 2 * Padding), value);
  }
  void fill(const T &non_padding_value, const T &padding_value) {
    for (int i = 0; i < m_width + 2 * Padding; i++) {
      for (int j = 0; j < m_height + 2 * Padding; j++) {
        if (i < Padding || i >= m_width + Padding ||
            j < Padding || j >= m_height + Padding) {
          (*this)(i, j) = padding_value;
        } else {
          (*this)(i, j) = non_padding_value;
        }
      }
    }
  }
  void swap(GhostArray2D &other) {
    m_data.swap(other.m_data);
    std::swap(m_width, other.m_width);
    std::swap(m_height, other.m_height);
  }
  void clear() {
    m_data.clear();
    m_width = 0;
    m_height = 0;
  }
  void reserve(int n) { m_data.reserve(n); }
  void shrink_to_fit() { m_data.shrink_to_fit(); }
  T *data() { return m_data.data(); }
  const T *data() const { return m_data.data(); }
  std::vector<T> &raw() { return m_data; }
  const std::vector<T> &raw() const { return m_data; }
 private:
  std::vector<T> m_data;
  int m_width = 0, m_height = 0;
};

template<typename T, int Padding, PaddingMode = PaddingMode::Around>
class GhostArray3D {
 public:
  GhostArray3D() = default;
  GhostArray3D(int width, int height, int depth)
      : m_width(width), m_height(height), m_depth(depth) {
    m_data.resize((width + 2 * Padding) * (height + 2 * Padding) *
        (depth + 2 * Padding));
  }
  T &operator()(int i, int j, int k) {
    return m_data[(i + Padding) +
        (j + Padding) * (m_width + 2 * Padding) +
        (k + Padding) * (m_width + 2 * Padding) *
            (m_height + 2 * Padding)];
  }
  const T &operator()(int i, int j, int k) const {
    return m_data[(i + Padding) +
        (j + Padding) * (m_width + 2 * Padding) +
        (k + Padding) * (m_width + 2 * Padding) *
            (m_height + 2 * Padding)];
  }
  T &operator[](int i) { return m_data[i]; }
  const T &operator[](int i) const { return m_data[i]; }
  int width() const { return m_width; }
  int height() const { return m_height; }
  int depth() const { return m_depth; }
  int size() const { return m_data.size(); }
  void resize(int width, int height, int depth) {
    m_width = width;
    m_height = height;
    m_depth = depth;
    m_data.resize((width + 2 * Padding) * (height + 2 * Padding) *
        (depth + 2 * Padding));
  }
  void resize(int width, int height, int depth, const T &value) {
    m_width = width;
    m_height = height;
    m_depth = depth;
    m_data.resize((width + 2 * Padding) * (height + 2 * Padding) *
        (depth + 2 * Padding), value);
  }
  void fill(const T &value) { std::fill(m_data.begin(), m_data.end(), value); }
  void swap(GhostArray3D &other) {
    m_data.swap(other.m_data);
    std::swap(m_width, other.m_width);
    std::swap(m_height, other.m_height);
    std::swap(m_depth, other.m_depth);
  }
  void clear() {
    m_data.clear();
    m_width = 0;
    m_height = 0;
    m_depth = 0;
  }
  void reserve(int n) { m_data.reserve(n); }
  void shrink_to_fit() { m_data.shrink_to_fit(); }
  T *data() { return m_data.data(); }
  const T *data() const { return m_data.data(); }
  std::vector<T> &raw() { return m_data; }
  const std::vector<T> &raw() const { return m_data; }
 private:
  std::vector<T> m_data;
  int m_width = 0, m_height = 0, m_depth = 0;
};
template<typename T>
using Array2D = GhostArray2D<T, 0>;
template<typename T>
using Array3D = GhostArray3D<T, 0>;

}

#endif //SIMCRAFT_CORE_INCLUDE_CORE_DATA_STRUCTURES_ARRAYS_H_
