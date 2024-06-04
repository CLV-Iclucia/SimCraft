//
// Created by creeper on 5/28/24.
//

#ifndef SIMCRAFT_FEM_INCLUDE_FEM_IPC_CONSTRAINT_H_
#define SIMCRAFT_FEM_INCLUDE_FEM_IPC_CONSTRAINT_H_
#include <fem/types.h>
namespace fem::ipc {
struct System;
struct Constraint {
  System *system;
  enum Type {
    EdgeEdge,
    VertexTriangle
  } type;
  // if type is EdgeEdge, ia and ib are the indices of the two edges
  // if type is VertexTriangle, ia is the index of the vertex and ib is the index of the triangle
  int ia, ib;
};
}
#endif //SIMCRAFT_FEM_INCLUDE_FEM_IPC_CONSTRAINT_H_
