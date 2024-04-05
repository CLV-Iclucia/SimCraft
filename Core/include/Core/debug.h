//
// Created by creeper on 24-3-20.
//

#ifndef DEBUG_H
#define DEBUG_H
#include <iostream>
#define ERROR(msg) \
  do { \
    std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << " " << msg << std::endl; \
    std::exit(1); \
  } while (0)

#endif //DEBUG_H
