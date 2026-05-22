#ifndef SPATIFY_PARALLEL_H_
#define SPATIFY_PARALLEL_H_

#ifdef USE_TBB
#include <tbb/tbb.h>
#endif
namespace spatify {
template <typename Func>
#ifdef USE_TBB
void parallel_for(int begin, int end, Func&& func) {
  tbb::parallel_for(begin, end, func);
}
template <typename Comp>
void parallel_sort(int begin, int end, Comp&& cmp) {
  tbb::parallel_sort(begin, end, cmp);
}
#else
template <typename Func>
void parallel_for(int begin, int end, Func&& func) {
  for (int i = begin; i < end; i++) {
    func(i);
  }
}
#endif
}
#endif