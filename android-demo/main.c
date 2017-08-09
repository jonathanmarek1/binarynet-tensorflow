#include <stdint.h>
#include <string.h>

#include <jni.h>
#include <android/log.h>
#include <android/native_window_jni.h>

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#define EGL_EGLEXT_PROTOTYPES
#include <EGL/eglext.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

#include "shaders.h"

#define printf(args...) __android_log_print(ANDROID_LOG_ERROR, "test_app", args)

#define jni(ret, name, args...) \
    JNIEXPORT ret JNICALL Java_test_app_MainActivity_ ## name(JNIEnv *env, jobject this, args)

#define jni0(ret, name) \
    JNIEXPORT ret JNICALL Java_test_app_MainActivity_ ## name(JNIEnv *env, jobject this)

struct egl {
    EGLDisplay display;
    EGLContext context;
    EGLConfig  config;
};

struct gl {
    GLuint program, program_oes;
    GLuint pos, texture, texture_matrix;
    GLuint pos_oes, texture_oes, texture_matrix_oes;
    GLuint camera_texture, fbo_texture;
    GLuint fbo;
};

uint64_t t0;
int paused;

#include <pthread.h>

uint8_t buf[227*227*4];
char out_string[1024] = "nothing.. yet";
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_out = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;

static uint64_t get_time(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

void egl_init(struct egl *egl)
{
    const EGLint attrib_list[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    const EGLint attribs[] = { EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
                               EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
                               EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8,
                               EGL_RED_SIZE, 8, EGL_ALPHA_SIZE, 8,
                               EGL_DEPTH_SIZE, 0, EGL_NONE };

    EGLDisplay display;
    EGLContext context;
    EGLConfig  config;
    EGLint count;

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    assert(display);

    if (!eglInitialize(display, 0, 0))
        assert(0);

    if (!eglChooseConfig(display, attribs, &config, 1, &count) || !count)
        assert(0);

    context = eglCreateContext(display, config, 0, attrib_list);
    assert(context);

    *egl = (struct egl) {display, context, config};
}

void gl_init(struct egl *egl, struct gl *gl, EGLSurface surface)
{
    if (!eglMakeCurrent(egl->display, surface, surface, egl->context))
        assert(0);

    gl->program = compile_program(vertex_shader, fragment_shader);
    gl->program_oes = compile_program(vertex_shader, fragment_shader_oes);
    assert(gl->program && gl->program_oes);

    gl->pos = glGetAttribLocation(gl->program, "pos");
    gl->texture = glGetUniformLocation(gl->program, "tex");
    gl->texture_matrix = glGetUniformLocation(gl->program, "texmat");

    gl->pos_oes = glGetAttribLocation(gl->program_oes, "pos");
    gl->texture_oes = glGetUniformLocation(gl->program_oes, "tex");
    gl->texture_matrix_oes = glGetUniformLocation(gl->program_oes, "texmat");

    glGenTextures(1, &gl->camera_texture);
    glGenFramebuffers(1, &gl->fbo);
    glGenTextures(1, &gl->fbo_texture);

    glBindTexture(GL_TEXTURE_2D, gl->fbo_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 227, 227, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, gl->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           gl->fbo_texture, 0);

    assert(glGetError() == 0);
}

static void matmul(float *r, float *a, float *b)
{
    int i, j, m;
    float sum;

    for (i = 0; i < 4; i++) for (j = 0; j < 4; j++) {
        sum = 0.0f;
        for (m = 0; m < 4; m++) {
            sum += b[i * 4 + m] * a[m * 4 + j];
        }
        r[i * 4 + j] = sum;
    }
}

void draw(struct egl *egl, struct gl *gl, float *mtx,
    EGLSurface surface, EGLSurface enc_surface)
{
    float matrix[16];

    memcpy(matrix, mtx, sizeof(matrix));

    int width, height;
    float vert[] = {-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f};

    if (enc_surface) {
        if (!eglMakeCurrent(egl->display, enc_surface, enc_surface, egl->context))
            assert(0);

        eglQuerySurface(egl->display, enc_surface, EGL_WIDTH, &width);
        eglQuerySurface(egl->display, enc_surface, EGL_HEIGHT, &height);
        glViewport(0, 0, width, height);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, gl->camera_texture);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glUseProgram(gl->program_oes);
        glUniform1i(gl->texture_oes, 0);

        glVertexAttribPointer(gl->pos_oes, 2, GL_FLOAT, GL_FALSE, 0, vert);
        glEnableVertexAttribArray(gl->pos_oes);

        glUniformMatrix4fv(gl->texture_matrix_oes, 1, 1, matrix);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        eglSwapBuffers(egl->display, enc_surface);
    }

    if (surface) {
        if (!eglMakeCurrent(egl->display, surface, surface, egl->context))
            assert(0);

        eglQuerySurface(egl->display, surface, EGL_WIDTH, &width);
        eglQuerySurface(egl->display, surface, EGL_HEIGHT, &height);
        glViewport(0, 0, width, height);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, gl->camera_texture);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        glUseProgram(gl->program_oes);
        glUniform1i(gl->texture_oes, 0);

        glVertexAttribPointer(gl->pos_oes, 2, GL_FLOAT, GL_FALSE, 0, vert);
        glEnableVertexAttribArray(gl->pos_oes);

        glUniformMatrix4fv(gl->texture_matrix_oes, 1, 1, matrix);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);


        float m = (float) width / (float) height;
        matmul(matrix, mtx, (float[]) {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f,    m, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, (1.0f - m)/2.0f, 0.0f, 1.0f,
        });

        glUniformMatrix4fv(gl->texture_matrix_oes, 1, 1, matrix);
        glBindFramebuffer(GL_FRAMEBUFFER, gl->fbo);
        glViewport(0, 0, 227, 227);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        pthread_mutex_lock(&mutex);
        glReadPixels(0, 0, 227, 227, GL_RGBA, GL_UNSIGNED_BYTE, buf);
        pthread_mutex_unlock(&mutex);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        glViewport(0, 0, width, height);

        float dx = 227.0f * 2.0f / width;
        float dy = 227.0f * 2.0f / height;
        float vert2[] = {-1.0f, -1.0f,
                        -1.0f, -1.0f + dy,
                        -1.0f + dx, -1.0f,
                        -1.0f + dx, -1.0f + dy};

        glUseProgram(gl->program);
        glUniform1i(gl->texture, 0);
        glBindTexture(GL_TEXTURE_2D, gl->fbo_texture);

        glVertexAttribPointer(gl->pos, 2, GL_FLOAT, GL_FALSE, 0, vert2);
        glEnableVertexAttribArray(gl->pos);


        glUniformMatrix4fv(gl->texture_matrix, 1, 1, (float[]) {
            2.0f / dx, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f / dy, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f,
        });
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        eglSwapBuffers(egl->display, surface);
    }
}

#include "../util.h"

ANativeWindow *window, *enc_window;
EGLSurface surface, enc_surface;
int camera_texture;
struct egl egl;
struct gl gl;

int network_id;

string weights, weights_bwn;

pthread_t thread;

#include "../xnornet_bwn.h"
#include "../xnornet.h"

#include "../rpi-demo/names2.h"

void* work_thread(void *arg)
{
    static float buf_in[227*227*3] __attribute__ ((aligned(16)));
    static uint8_t tmpbuf[xnornet_bwn_tmp_size] __attribute__ ((aligned(16))); // TODO
    float *y;
    int i, j, top[5], id;

    while (1) {
        pthread_mutex_lock(&mutex);
        if (paused)
            pthread_cond_wait(&cond, &mutex);

        for (i = 0; i < 227; i++) for (j = 0; j < 227; j++) {
            float m[] = {0.01735949, 0.01772787, 0.01774145};
            float b[] = {-2.13645733, -2.04468092, -1.81410977};
            buf_in[i*227*3+(226-j)*3+0] = (float) buf[j*227*4+i*4+0] * m[0] + b[0];
            buf_in[i*227*3+(226-j)*3+1] = (float) buf[j*227*4+i*4+1] * m[1] + b[1];
            buf_in[i*227*3+(226-j)*3+2] = (float) buf[j*227*4+i*4+2] * m[2] + b[2];
        }

        id = network_id;

        pthread_mutex_unlock(&mutex);

        if (network_id == 0)
            y = xnornet_bwn(buf_in, weights_bwn.ptr, tmpbuf);
        else
            y = xnornet(buf_in, weights.ptr, tmpbuf);

        softmax(y, 1000);
        top5(top, y, 1000);

        pthread_mutex_lock(&mutex_out);

        sprintf(out_string,"Network: %s\n%s:%f\n%s:%f\n%s:%f\n%s:%f\n%s:%f\n",
            (char*[]) {"BWN","XNORNET"}[network_id],
            names[top[0]], y[top[0]],
            names[top[1]], y[top[1]],
            names[top[2]], y[top[2]],
            names[top[3]], y[top[3]],
            names[top[4]], y[top[4]]);

        pthread_mutex_unlock(&mutex_out);
    }
}

jni(int, init, jobject _surface)
{
    //EGLSurface surface;
    //ANativeWindow *window;
    int ret;

    printf("init\n");
    egl_init(&egl);

    /*window = ANativeWindow_fromSurface(env, _surface);
    assert(window);
    surface = eglCreateWindowSurface(egl.display, egl.config, window, 0);
    assert(surface); */

    enc_window = ANativeWindow_fromSurface(env, _surface);
    assert(enc_window);
    enc_surface = eglCreateWindowSurface(egl.display, egl.config, enc_window, 0);
    assert(enc_surface);

    gl_init(&egl, &gl, enc_surface);
    /*eglMakeCurrent(egl.display, 0, 0, 0);

    eglDestroySurface(egl.display, surface);
    ANativeWindow_release(window);*/

    ret = file_mmap(&weights_bwn, "/sdcard/xnornet_bwn_weights");
    assert(!ret && weights_bwn.size == xnornet_bwn_size);

    ret = file_mmap(&weights, "/sdcard/xnornet_weights");
    assert(!ret && weights.size == xnornet_size);

    t0 = get_time();

    pthread_create(&thread, 0, work_thread, 0);

    return gl.camera_texture;
}

jni0(int, exit)
{
    return 0;
}

jni(void, draw, jfloatArray mtx)
{
    float *m;

    m = (*env)->GetFloatArrayElements(env, mtx, 0);

    //assert(enc_window);
    draw(&egl, &gl, m, surface, 0);
    //eglMakeCurrent(egl.display, 0, 0, 0);

    (*env)->ReleaseFloatArrayElements(env, mtx, m, 0);
}

jni(void, encode, jfloatArray mtx)
{
    float *m;

    m = (*env)->GetFloatArrayElements(env, mtx, 0);

    draw(&egl, &gl, m, 0, enc_surface);

    (*env)->ReleaseFloatArrayElements(env, mtx, m, 0);
}

jni0(jstring, getoverlay)
{
    jstring out;
    pthread_mutex_lock(&mutex_out);
    out = (*env)->NewStringUTF(env, out_string);
    pthread_mutex_unlock(&mutex_out);
    return out;
}

jni(void, setcodecsurface, jobject _surface)
{
    if (enc_window) {
        eglMakeCurrent(egl.display, 0, 0, 0);
        eglDestroySurface(egl.display, enc_surface);
        ANativeWindow_release(enc_window);
    }

    enc_window = ANativeWindow_fromSurface(env, _surface);
    assert(enc_window);
    enc_surface = eglCreateWindowSurface(egl.display, egl.config, enc_window, 0);
    assert(enc_surface);

    eglMakeCurrent(egl.display, enc_surface, enc_surface, egl.context);
}

void set_paused(int pause)
{
    pthread_mutex_lock(&mutex);
    paused = pause;
    if (!pause)
        pthread_cond_broadcast(&cond);
    pthread_mutex_unlock(&mutex);
}

jni(void, created, jobject _surface)
{
    set_paused(0);

    window = ANativeWindow_fromSurface(env, _surface);
    assert(window);

    printf("surface %p->%p\n", _surface, window);

    surface = eglCreateWindowSurface(egl.display, egl.config, window, 0);
    assert(surface);
}

jni(void, changed, jobject surface, int format, int width, int height)
{
    // does anything need to be done with this?
}

jni(void, destroyed, jobject null)
{
    set_paused(1);

    eglDestroySurface(egl.display, surface);
    ANativeWindow_release(window);

    window = 0;
    surface = 0;
}

jni(void, setnetwork, jint id)
{
    pthread_mutex_lock(&mutex);
    network_id = id;
    pthread_mutex_unlock(&mutex);
}

/* keep a lists of frames which start with keyframes to create clips
   these can then be muxed to create a video file
   also keep codec configuration data to use as starting information
        *uses the last recieved configuration so may be wrong in some cases
*/

#include <media/NdkMediaMuxer.h>
#define FLAG_KEY_FRAME 1
#define FLAG_CODEC_CONFIG 2

struct frames {
    size_t num_frame, data_size;
    void *data;
    struct AMediaCodecBufferInfo *info;
};

uint8_t codec_config[256];
int codec_config_size;
int64_t last_timestamp;

#define RING_SIZE 8 // currently used as lazy way to determine clip length

struct {
    int first, last;
    struct frames data[RING_SIZE];
} frames;

jni(void, addbuffer, jobject buf, int offset, int size, int64_t timestamp, int flags)
{
    struct frames *f;
    void *ptr;
    int i;

    last_timestamp = timestamp;

    ptr = (*env)->GetDirectBufferAddress(env, buf);
    assert(ptr);

    assert(flags == FLAG_KEY_FRAME || flags == FLAG_CODEC_CONFIG || !flags);

    if (flags == FLAG_CODEC_CONFIG) {
        assert(size < 256);
        memcpy(codec_config, ptr + offset, size);
        codec_config_size = size;
        // clear everything
        for (i = frames.first;; ) {
            f = &frames.data[i];
            free(f->data);
            free(f->info);
            f->data = 0;
            f->info = 0;
            f->num_frame = 0;
            f->data_size = 0;
            if (i == frames.last)
                break;
            i = (i + 1) % RING_SIZE;
        }
        frames.first = frames.last;
        return;
    }

    if (flags == FLAG_KEY_FRAME) {
        frames.last = (frames.last + 1) % RING_SIZE;
        if (frames.last == frames.first) {
            f = &frames.data[frames.first];
            free(f->data);
            free(f->info);
            f->data = 0;
            f->info = 0;
            f->num_frame = 0;
            f->data_size = 0;
            frames.first = (frames.first + 1) % RING_SIZE;
        }
    }

    f = &frames.data[frames.last];

    f->data = realloc(f->data, f->data_size + size);
    assert(f->data);
    f->info = realloc(f->info, (f->num_frame+1) * sizeof(*f->info));
    assert(f->info);

    f->info[f->num_frame] = (struct AMediaCodecBufferInfo) {f->data_size, size, timestamp, flags};
    memcpy(f->data + f->data_size, ptr + offset, size);

    f->num_frame += 1;
    f->data_size += size;
}

jni(void, writemux, jstring path)
{
    AMediaMuxer *mux;
    ssize_t track;
    struct frames *f;
    int i, j, fd;

    const char *path_c = (*env)->GetStringUTFChars(env, path, 0);
    assert(path_c);

    fd = open(path_c, O_WRONLY | O_CREAT, 0666);
    assert(fd >= 0);

    (*env)->ReleaseStringUTFChars(env, path, path_c);

    mux = AMediaMuxer_new(fd, AMEDIAMUXER_OUTPUT_FORMAT_MPEG_4);
    assert(mux);

    {
        AMediaFormat *fmt;

        fmt = AMediaFormat_new();

        AMediaFormat_setString(fmt, AMEDIAFORMAT_KEY_MIME, "video/avc");

        //
        AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, 640);
        AMediaFormat_setInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, 480);

        // native media muxer doesnt support codec config sample data..
        // hack around it

        int split = -1;
        for (i = 4; i < codec_config_size - 3; i++) {
            if (!memcmp(codec_config + i, (char[]) {0, 0, 0, 1}, 4)) {
                split = i;
                break;
            }
        }
        assert(split >= 0);

        AMediaFormat_setBuffer(fmt, "csd-0", codec_config, split);
        AMediaFormat_setBuffer(fmt, "csd-1", codec_config + split, codec_config_size - split);

        track = AMediaMuxer_addTrack(mux, fmt);
        assert(track >= 0);

        AMediaFormat_delete(fmt);
    }

    AMediaMuxer_start(mux);

    //AMediaMuxer_writeSampleData(mux, track, codec_config,
    //    &(struct AMediaCodecBufferInfo) {0, codec_config_size, 0, FLAG_CODEC_CONFIG});

    for (i = frames.first;; ) {
        f = &frames.data[i];
        printf("id %i %i\n", i, f->num_frame);
        for (j = 0; j < f->num_frame; j++)
            AMediaMuxer_writeSampleData(mux, track, f->data, &f->info[j]);
        if (i == frames.last)
            break;
        i = (i + 1) % RING_SIZE;
    }

    AMediaMuxer_stop(mux);
    AMediaMuxer_delete(mux);
    close(fd);
}

