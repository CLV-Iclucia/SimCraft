// add header protection
#ifndef CORE_RENDER_BUFFER_OBJ_H
#define CORE_RENDER_BUFFER_OBJ_H

#include <glad/glad.h>
#include <vector>
#include <Core/core.h>
namespace core {
// RAII buffer object
struct VertexBufferObj {
    uint id; 
    VertexBufferObj() {
        glGenBuffers(1, &id);
        glBindBuffer(GL_ARRAY_BUFFER, id);
    }
    template <typename T>
    void passData(const std::vector<T>& data) {
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), data.data(), GL_STATIC_DRAW);
    }
    template <typename T>
    void passData(const T* data, int size) {
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), data, GL_STATIC_DRAW);
    }
    void bind() {
        glBindBuffer(GL_ARRAY_BUFFER, id);
    }
    ~VertexBufferObj() {
        glDeleteBuffers(1, &id);
    }
};

struct VertexArrayObj {
    uint id;
    VertexArrayObj() {
        glGenVertexArrays(1, &id);
    }
    void bind() {
        glBindVertexArray(id);
    }
    // methods for attributes

    void unbind() {
        glBindVertexArray(0);
    }
    ~VertexArrayObj() {
        glDeleteVertexArrays(1, &id);
    }
};

struct ElementBufferObj {
    uint id;
    ElementBufferObj() {
        glGenBuffers(1, &id);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
    }
    ElementBufferObj(const std::vector<uint>& data) {
        glGenBuffers(1, &id);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(uint), data.data(), GL_STATIC_DRAW);
    }
    void passData(const std::vector<uint>& data) {
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, data.size() * sizeof(uint), data.data(), GL_STATIC_DRAW);
    }
    void bind() {
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, id);
    }
    ~ElementBufferObj() {
        glDeleteBuffers(1, &id);
    }
};
}


#endif 