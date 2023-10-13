#ifndef CORE_SHADER_PROG_H_
#define CORE_SHADER_PROG_H_

#include <string>
#include <glad/glad.h>
#include <fstream>
#include <sstream>
#include <iostream>
namespace Core {
// RAII shader program
struct ShaderProg {
    static char error_log[512];
    uint id;
    // create shader program from vertex and fragment shader source
    ShaderProg(const std::string& vs_path, const std::string& fs_path) {
        id = glCreateProgram();
        uint vert_id = glCreateShader(GL_VERTEX_SHADER);
        uint frag_id = glCreateShader(GL_FRAGMENT_SHADER);
        // read shader source using fstream
        std::string vert, frag;
        std::ifstream vs_file, fs_file;
        vs_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        fs_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try {
            vs_file.open(vs_path);
            fs_file.open(fs_path);
            std::stringstream vs_stream, fs_stream;
            vs_stream << vs_file.rdbuf();
            fs_stream << fs_file.rdbuf();
            vs_file.close();
            fs_file.close();
            vert = vs_stream.str();
            frag = fs_stream.str();
        } catch (std::ifstream::failure e) {
            std::cerr << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ" << std::endl;
        }
        const char* vert_src = vert.c_str();
        const char* frag_src = frag.c_str();
        glShaderSource(vert_id, 1, &vert_src, NULL);
        glShaderSource(frag_id, 1, &frag_src, NULL);
        glCompileShader(vert_id);
        glCompileShader(frag_id);
        glAttachShader(id, vert_id);
        glAttachShader(id, frag_id);
        glLinkProgram(id);
        glDeleteShader(vert_id);
        glDeleteShader(frag_id);
    }
    void use() {
        glUseProgram(id);
    }
    void setInt(const std::string& name, int value) {
        glUniform1i(glGetUniformLocation(id, name.c_str()), value);
    }
    void setFloat(const std::string& name, float value) {
        glUniform1f(glGetUniformLocation(id, name.c_str()), value);
    }
    void setVec2(const std::string& name, float x, float y) {
        glUniform2f(glGetUniformLocation(id, name.c_str()), x, y);
    } 
    ~ShaderProg() {
        glDeleteProgram(id);
    }
};
}

#endif