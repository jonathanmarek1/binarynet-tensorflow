#include <GLES2/gl2.h>
#include <assert.h>

char *vertex_shader =
    "attribute vec2 pos;\n"
    "uniform vec4 texmat;\n"
    "varying vec2 coord;\n"
    "void main() {\n"
    "  coord = pos * texmat.xy + texmat.zw;\n"
    "  gl_Position = vec4(pos, 0.0, 1.0);\n"
    "}\n";

char *fragment_shader =
    "precision mediump float;\n"
    "varying vec2 coord;\n"
    "uniform sampler2D tex;\n"
    "void main() {\n"
    "  gl_FragColor = vec4(texture2D(tex, coord).rgb, 1.0);\n"
    "}\n";

char *fragment_shader_oes =
    "#extension GL_OES_EGL_image_external : require\n"
    "precision mediump float;\n"
    "varying vec2 coord;\n"
    "uniform samplerExternalOES tex;\n"
    "void main() {\n"
    "  gl_FragColor = vec4(texture2D(tex, coord).rgb, 1.0);\n"
    "}\n";

/*char* ver =
    "attribute vec2 vPosition;\n"
    "varying vec2 coord;\n"
    "void main() {\n"
    "  coord = (vPosition * vec2(1.0, 1.0) + vec2(1.0, 1.0)) * 0.5;\n"
    "  gl_Position = vec4((vPosition + vec2(1.0, 1.0)) * vec2(227.0 / 1920.0, 227.0 / 1080.0) - vec2(1.0, 1.0), 0.0, 1.0);\n"
    "}\n"; */


static GLuint compile_shader(GLenum type, const char* source)
{
    GLuint shader;
    GLint status;

    shader = glCreateShader(type);
    if (!shader)
        return 0;

    glShaderSource(shader, 1, &source, 0);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (!status) {
        glDeleteShader(shader);
        assert(0);
    }

    return shader;
}

GLuint compile_program(char *vertex_source, char *fragment_source)
{
    GLuint vertex_shader, fragment_shader, program;
    GLint status;

    vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source);
    fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    assert(vertex_shader && fragment_shader);

    program = glCreateProgram();
    assert(program);

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &status);
    if (!status)
        assert(0);

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}
