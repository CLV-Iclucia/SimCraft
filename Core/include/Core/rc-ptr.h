#pragma once

namespace ksk::core {

template <class T>
class rc_ptr {
 public:
  rc_ptr() = default;

  explicit rc_ptr(T* ptr) : ptr_(ptr)
  {
    if (ptr_) {
      ptr_->addRef();
    }
  }

  rc_ptr(const rc_ptr& other) : ptr_(other.ptr_)
  {
    if (ptr_) {
      ptr_->addRef();
    }
  }

  rc_ptr(rc_ptr&& other) noexcept : ptr_(other.ptr_)
  {
    other.ptr_ = nullptr;
  }

  ~rc_ptr()
  {
    if (ptr_) {
      ptr_->release();
    }
  }

  rc_ptr& operator=(const rc_ptr& other)
  {
    if (this != &other) {
      if (other.ptr_) {
        other.ptr_->addRef();
      }
      if (ptr_) {
        ptr_->release();
      }
      ptr_ = other.ptr_;
    }
    return *this;
  }

  rc_ptr& operator=(rc_ptr&& other) noexcept
  {
    if (this != &other) {
      if (ptr_) {
        ptr_->release();
      }
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }

  T* operator->() const noexcept { return ptr_; }
  T& operator*() const noexcept { return *ptr_; }
  T* get() const noexcept { return ptr_; }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }
  [[nodiscard]] bool valid() const noexcept { return ptr_ != nullptr; }

  void reset() noexcept
  {
    if (ptr_) {
      ptr_->release();
      ptr_ = nullptr;
    }
  }

  static rc_ptr adopt(T* raw) noexcept
  {
    rc_ptr result;
    result.ptr_ = raw;
    return result;
  }

  friend bool operator==(const rc_ptr& a, const rc_ptr& b) noexcept
  {
    return a.ptr_ == b.ptr_;
  }

  friend bool operator!=(const rc_ptr& a, const rc_ptr& b) noexcept
  {
    return a.ptr_ != b.ptr_;
  }

 private:
  T* ptr_ = nullptr;
};

}  // namespace ksk::core
