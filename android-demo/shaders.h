#include <GLES2/gl2.h>

extern char *vertex_shader, *fragment_shader, *fragment_shader_oes;

GLuint compile_program(char *vertex_source, char *fragment_source);
