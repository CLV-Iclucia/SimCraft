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
    void allocData(const std::vector<T>& data) {
        glBufferData(GL_ARRAY_BUFFER, data.size() * sizeof(T), (void*)data.data(), GL_STATIC_DRAW);
    }
    template <typename T>
    void allocData(const T* data, int size) {
        glBufferData(GL_ARRAY_BUFFER, size * sizeof(T), (void*)data, GL_STATIC_DRAW);
    }
    template <typename T>
    void passData(const std::vector<T>& data) {
        glBufferSubData(GL_ARRAY_BUFFER, 0, data.size() * sizeof(T), (void*)data.data());
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

    static void unbind() {
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