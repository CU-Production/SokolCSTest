// based on iq's raymarching demo
// https://www.shadertoy.com/view/Xds3zN

#define SOKOL_IMPL
#define SOKOL_NO_ENTRY
#define SOKOL_GLCORE
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

#include "HandmadeMath.h"

#include <vector>
#include <fstream>
#include <iostream>

constexpr uint32_t SCREEN_WIDTH = 800;
constexpr uint32_t SCREEN_HEIGHT = 600;

struct cs_params_t{
    HMM_Vec2 iTime;
    HMM_Vec2 iResolution;
    HMM_Vec4 iMouse;
};

struct particle_t{
    HMM_Vec2 pos;
    HMM_Vec2 vel;
    HMM_Vec4 color;
};

struct {
    struct {
        sg_image img;
        sg_attachments atts;
        sg_pipeline pip;
        cs_params_t params;
    } compute;
    struct {
        sg_pipeline pip;
        sg_pass_action pass_action;
        sg_sampler smp;
    } graphics;
} state;

void init() {
    sg_desc _sg_desc{};
    _sg_desc.environment = sglue_environment();
    _sg_desc.logger.func = slog_func;
    sg_setup(&_sg_desc);

    // compute
    {
        sg_image_desc _sg_image_desc{};
        _sg_image_desc.usage.storage_attachment = true;
        _sg_image_desc.width = SCREEN_WIDTH;
        _sg_image_desc.height = SCREEN_HEIGHT;
        _sg_image_desc.pixel_format = SG_PIXELFORMAT_RGBA8;
        _sg_image_desc.label = "noise-image";
        state.compute.img = sg_make_image(&_sg_image_desc);

        sg_attachments_desc _sg_attachments_desc{};
        _sg_attachments_desc.storages[0].image = state.compute.img;
        _sg_attachments_desc.label = "noise-attachments";
        state.compute.atts = sg_make_attachments(&_sg_attachments_desc);

        std::ifstream file("raymarching_gl.glsl", std::ios::ate);
        if (!file.is_open()) {
            std::cerr << "Could not open file " << __FILE__ << " at line " << __LINE__ << std::endl;
            std::exit(1);
        }
        size_t file_size = (size_t)file.tellg();
        std::vector<char> file_content(file_size);
        file.seekg(0);
        file.read(file_content.data(), file_size);
        file.close();

        sg_shader_desc _sg_compute_shader_desc{};
        _sg_compute_shader_desc.compute_func.source = file_content.data();

        _sg_compute_shader_desc.uniform_blocks[0].stage = SG_SHADERSTAGE_COMPUTE;
        _sg_compute_shader_desc.uniform_blocks[0].size = sizeof(cs_params_t);
        _sg_compute_shader_desc.uniform_blocks[0].glsl_uniforms[0] = { .type = SG_UNIFORMTYPE_FLOAT2, .glsl_name = "iTime",  };
        _sg_compute_shader_desc.uniform_blocks[0].glsl_uniforms[1] = { .type = SG_UNIFORMTYPE_FLOAT2, .glsl_name = "iResolution",  };
        _sg_compute_shader_desc.uniform_blocks[0].glsl_uniforms[2] = { .type = SG_UNIFORMTYPE_FLOAT4, .glsl_name = "iMouse",  };

        _sg_compute_shader_desc.storage_images[0].stage = SG_SHADERSTAGE_COMPUTE;
        _sg_compute_shader_desc.storage_images[0].image_type = SG_IMAGETYPE_2D;
        _sg_compute_shader_desc.storage_images[0].access_format = SG_PIXELFORMAT_RGBA8;
        _sg_compute_shader_desc.storage_images[0].writeonly = true;
        _sg_compute_shader_desc.storage_images[0].glsl_binding_n = 0;

        _sg_compute_shader_desc.label = "compute-shader";

        sg_shader compute_shd = sg_make_shader(&_sg_compute_shader_desc);

        sg_pipeline_desc _compute_pipeline_desc{};
        _compute_pipeline_desc.compute = true;
        _compute_pipeline_desc.shader = compute_shd;
        _compute_pipeline_desc.label = "compute-pipeline";

        state.compute.pip = sg_make_pipeline(&_compute_pipeline_desc);

        state.compute.params = { {0.0f, 0.0f}, {SCREEN_WIDTH, SCREEN_HEIGHT}, {0.0f, 0.0f, 0.0f, 0.0f}};
    }

    // graphics
    {
        sg_shader_desc _shader_desc{};
        _shader_desc.vertex_func.source = R"(
#version 430 core
layout(location=0) out vec2 vUV;

const vec4 vertices[4] = {
  // pos         uv
  {-1.0f, -1.0f, 0.0f, 1.0f},
  { 1.0f, -1.0f, 1.0f, 1.0f},
  {-1.0f,  1.0f, 0.0f, 0.0f},
  { 1.0f,  1.0f, 1.0f, 0.0f},
};
const int indices[6] = { 0, 1, 2, 1, 3, 2 };

void main() {
  vec2 position = vertices[indices[gl_VertexID]].xy;
  vec2 texcoord0 = vertices[indices[gl_VertexID]].zw;
  gl_Position = vec4(position, 0.0f, 1.0f);
  vUV = texcoord0;
}
)";
        _shader_desc.fragment_func.source = R"(
#version 430 core
layout(binding=0) uniform sampler2D disp_tex;
layout(location=0) in vec2 vUV;
out vec4 frag_color;

void main() {
  frag_color = vec4(texture(disp_tex, vUV).xyz, 1.0f);
}
)";

        _shader_desc.label = "fragment-shader";
        _shader_desc.images[0].stage = SG_SHADERSTAGE_FRAGMENT;
        _shader_desc.images[0].image_type = SG_IMAGETYPE_2D;
        _shader_desc.images[0].sample_type = SG_IMAGESAMPLETYPE_FLOAT;
        _shader_desc.samplers[0].stage = SG_SHADERSTAGE_FRAGMENT;
        _shader_desc.samplers[0].sampler_type = SG_SAMPLERTYPE_FILTERING;
        _shader_desc.image_sampler_pairs[0].stage = SG_SHADERSTAGE_FRAGMENT;
        _shader_desc.image_sampler_pairs[0].image_slot = 0;
        _shader_desc.image_sampler_pairs[0].sampler_slot = 0;
        _shader_desc.image_sampler_pairs[0].glsl_name = "disp_tex";

        sg_shader graphics_shd = sg_make_shader(&_shader_desc);

        sg_pipeline_desc _pipeline_desc{};
        _pipeline_desc.shader = graphics_shd;
        _pipeline_desc.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
        state.graphics.pip = sg_make_pipeline(&_pipeline_desc);

        state.graphics.pass_action.colors[0] = { .load_action=SG_LOADACTION_CLEAR, .clear_value={0.2f, 0.3f, 0.3f, 1.0f } };

        sg_sampler_desc _sg_sampler_desc{};
        _sg_sampler_desc.min_filter = SG_FILTER_LINEAR;
        _sg_sampler_desc.mag_filter = SG_FILTER_LINEAR;
        _sg_sampler_desc.label = "Linear sampler";
        state.graphics.smp = sg_make_sampler(&_sg_sampler_desc);
    }
 }

void frame() {
    const double dt = sapp_frame_duration();

    state.compute.params.iTime.X += (float)dt;
    state.compute.params.iTime.Y  = (float)dt;

    // compute pass
    sg_pass _compute_pass = { .compute=true, .attachments = state.compute.atts, .label="compute_pass" };
    sg_begin_pass(&_compute_pass);
    sg_apply_pipeline(state.compute.pip);
    sg_apply_uniforms(0, SG_RANGE(state.compute.params));
    sg_dispatch((SCREEN_WIDTH + 7)/8, (SCREEN_HEIGHT + 7)/8, 1);
    sg_end_pass();

    // graphics pass
    sg_bindings _graphics_bindings{};
    _graphics_bindings.images[0] = state.compute.img;
    _graphics_bindings.samplers[0] = state.graphics.smp;
    sg_pass _graphics_pass = { .action=state.graphics.pass_action, .swapchain=sglue_swapchain(), .label="render-pass"  };
    sg_begin_pass(&_graphics_pass);
    sg_apply_pipeline(state.graphics.pip);
    sg_apply_bindings(_graphics_bindings);
    sg_draw(0, 6, 1);
    sg_end_pass();
    sg_commit();
}

void cleanup() {
    sg_shutdown();
}

void input(const sapp_event* event) {
    static bool left_button_has_clicked = false;
    switch (event->type) {
        case SAPP_EVENTTYPE_MOUSE_DOWN: {
            if (event->mouse_button == SAPP_MOUSEBUTTON_LEFT) {
                state.compute.params.iMouse.XY = {event->mouse_x, event->mouse_y};
                left_button_has_clicked = true;
            }
            state.compute.params.iMouse.Z = event->mouse_button == SAPP_MOUSEBUTTON_LEFT;
            state.compute.params.iMouse.W = event->mouse_button == SAPP_MOUSEBUTTON_RIGHT;
            break;
        }
        case SAPP_EVENTTYPE_MOUSE_UP: {
            if (event->mouse_button == SAPP_MOUSEBUTTON_LEFT) {
                left_button_has_clicked = false;
            }
            break;
        }
        case SAPP_EVENTTYPE_MOUSE_MOVE: {
            if (left_button_has_clicked) {
                state.compute.params.iMouse.XY = {event->mouse_x, event->mouse_y};
            }
            break;
        }
        default: break;
    }
}

int main() {
    sapp_desc desc = {0};
    desc.init_cb = init;
    desc.frame_cb = frame;
    desc.cleanup_cb = cleanup,
    desc.event_cb = input,
    desc.width  = SCREEN_WIDTH,
    desc.height = SCREEN_HEIGHT,
    desc.window_title = "sokol cs noise (GL4.3)",
    desc.icon.sokol_default = true,
    desc.logger.func = slog_func;
    sapp_run(&desc);

    return 0;
}
