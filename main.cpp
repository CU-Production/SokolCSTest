#define SOKOL_IMPL
#define SOKOL_NO_ENTRY
#define SOKOL_GLCORE
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

constexpr uint32_t SCREEN_WIDTH = 800;
constexpr uint32_t SCREEN_HEIGHT = 600;

sg_pipeline pip;
sg_bindings bind;
sg_pass_action pass_action;

void init() {
    sg_desc _sg_desc{};
    _sg_desc.environment = sglue_environment();
    _sg_desc.logger.func = slog_func;
    sg_setup(&_sg_desc);

    float vertices[] = {
        // positions            // colors
        0.0f,  0.5f, 0.5f,     1.0f, 0.0f, 0.0f, 1.0f,
        0.5f, -0.5f, 0.5f,     0.0f, 1.0f, 0.0f, 1.0f,
       -0.5f, -0.5f, 0.5f,     0.0f, 0.0f, 1.0f, 1.0f
    };

    sg_buffer_desc _vertex_buffer_desc{};
    _vertex_buffer_desc.data = SG_RANGE(vertices);
    _vertex_buffer_desc.label = "triangle-vertices";
    bind.vertex_buffers[0] = sg_make_buffer(&_vertex_buffer_desc);

    sg_shader_desc _shader_desc{};
    _shader_desc.vertex_func.source = R"(
#version 430 core

layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
layout(location=0) out vec4 vColor;

void main() {
    gl_Position = vec4(aPos, 1.0f);
    vColor = aColor;
}
)";
    _shader_desc.fragment_func.source = R"(
#version 430 core
layout(location=0) in vec4 vColor;
out vec4 frag_color;

void main() {
    frag_color = vColor;
}
)";
    sg_shader shd = sg_make_shader(&_shader_desc);

    sg_pipeline_desc _pipeline_desc{};
    _pipeline_desc.shader = shd;
    _pipeline_desc.layout.attrs[0] = { .buffer_index = 0, .offset = 0,  .format = SG_VERTEXFORMAT_FLOAT3};
    _pipeline_desc.layout.attrs[1] = { .buffer_index = 0, .offset = 12, .format = SG_VERTEXFORMAT_FLOAT4};
    pip = sg_make_pipeline(&_pipeline_desc);

    pass_action.colors[0] = { .load_action=SG_LOADACTION_CLEAR, .clear_value={0.2f, 0.3f, 0.3f, 1.0f } };
 }

void frame() {
    const double dt = sapp_frame_duration();

    sg_pass _sg_pass = { .action=pass_action, .swapchain=sglue_swapchain() };
    sg_begin_pass(&_sg_pass);
    sg_apply_pipeline(pip);
    sg_apply_bindings(bind);
    sg_draw(0, 3, 1);
    sg_end_pass();
    sg_commit();
}

void cleanup() {
    sg_shutdown();
}

void input(const sapp_event* event) {}

int main() {
    sapp_desc desc = {0};
    desc.init_cb = init;
    desc.frame_cb = frame;
    desc.cleanup_cb = cleanup,
    desc.event_cb = input,
    desc.width  = SCREEN_WIDTH,
    desc.height = SCREEN_HEIGHT,
    desc.window_title = "sokol + puredoom",
    desc.icon.sokol_default = true,
    desc.logger.func = slog_func;
    sapp_run(&desc);

    return 0;
}
