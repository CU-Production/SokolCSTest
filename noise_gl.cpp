#define SOKOL_IMPL
#define SOKOL_NO_ENTRY
#define SOKOL_GLCORE
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

#include "HandmadeMath.h"

#include <vector>

constexpr uint32_t SCREEN_WIDTH = 800;
constexpr uint32_t SCREEN_HEIGHT = 600;

struct cs_params_t{
    float time;
    HMM_Vec2 img_size;
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
        sg_buffer vbuf;
        sg_buffer ibuf;
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

        sg_shader_desc _sg_compute_shader_desc{};
        _sg_compute_shader_desc.compute_func.source = R"(
#version 430
uniform float time;
uniform vec2 img_size;

layout(binding=0, rgba8) uniform writeonly image2D cs_out_tex;
layout(local_size_x=8, local_size_y=8, local_size_y=1) in;

float hash12(vec2 p)
{
  vec3 p3 = fract(vec3(p.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

void main() {
  uvec2 gid = gl_GlobalInvocationID.xy;
  if (gid.x >= img_size.x || gid.y > img_size.y) {
    return;
  }

  vec2 p = vec2(gl_GlobalInvocationID.xy + time);
  imageStore(cs_out_tex, ivec2(gl_GlobalInvocationID.xy), vec4(hash12(p), hash12(p + (0.1f).xx), hash12(p + (0.2f).xx), 1.0f));
}
)";

        _sg_compute_shader_desc.uniform_blocks[0].stage = SG_SHADERSTAGE_COMPUTE;
        _sg_compute_shader_desc.uniform_blocks[0].size = sizeof(cs_params_t);
        _sg_compute_shader_desc.uniform_blocks[0].glsl_uniforms[0] = { .type = SG_UNIFORMTYPE_FLOAT, .glsl_name = "time",  };
        _sg_compute_shader_desc.uniform_blocks[0].glsl_uniforms[1] = { .type = SG_UNIFORMTYPE_FLOAT2, .glsl_name = "img_size",  };

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

        state.compute.params = { 0.0f, {SCREEN_WIDTH, SCREEN_HEIGHT}};
    }

    // graphics
    {
        sg_shader_desc _shader_desc{};
        _shader_desc.vertex_func.source = R"(
#version 430 core

layout(location=0) in vec3 position;
layout(location=1) in vec2 texcoord0;

layout(location=0) out vec2 vUV;

void main() {
  gl_Position = vec4(position, 1.0f);
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
        _shader_desc.attrs[0].glsl_name = "position";
        _shader_desc.attrs[1].glsl_name = "texcoord0";
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
        _pipeline_desc.layout.attrs[0].format = SG_VERTEXFORMAT_FLOAT3;
        _pipeline_desc.layout.attrs[1].format = SG_VERTEXFORMAT_FLOAT2;
        _pipeline_desc.index_type = SG_INDEXTYPE_UINT32;
        state.graphics.pip = sg_make_pipeline(&_pipeline_desc);

        state.graphics.pass_action.colors[0] = {
            .load_action=SG_LOADACTION_CLEAR,
            .store_action=SG_STOREACTION_STORE,
            .clear_value={0.2f, 0.3f, 0.3f, 1.0f }
        };

        const float vertices[] = {
            // positions     uv
            -1.0, -1.0, 0.0, 0.0, 1.0,
            1.0,  -1.0, 0.0, 1.0, 1.0,
            1.0,  1.0,  0.0, 1.0, 0.0,
            -1.0, 1.0,  0.0, 0.0, 0.0,
        };
        sg_buffer_desc vb_desc = {};
        vb_desc.data = SG_RANGE(vertices);
        state.graphics.vbuf = sg_make_buffer(&vb_desc);

        const int indices[] = { 0, 1, 2, 0, 2, 3, };
        sg_buffer_desc ib_desc = {};
        ib_desc.usage.index_buffer = true;
        ib_desc.data = SG_RANGE(indices);
        state.graphics.ibuf = sg_make_buffer(&ib_desc);

        sg_sampler_desc _sg_sampler_desc{};
        _sg_sampler_desc.min_filter = SG_FILTER_LINEAR;
        _sg_sampler_desc.mag_filter = SG_FILTER_LINEAR;
        _sg_sampler_desc.label = "Linear sampler";
        state.graphics.smp = sg_make_sampler(&_sg_sampler_desc);
    }
 }

void frame() {
    const double dt = sapp_frame_duration();

    state.compute.params.time += (float)dt;

    // compute pass
    sg_pass _compute_pass = { .compute=true, .attachments = state.compute.atts, .label="compute_pass" };
    sg_begin_pass(&_compute_pass);
    sg_apply_pipeline(state.compute.pip);
    sg_apply_uniforms(0, SG_RANGE(state.compute.params));
    sg_dispatch((SCREEN_WIDTH + 7)/8, (SCREEN_HEIGHT + 7)/8, 1);
    sg_end_pass();

    // graphics pass
    sg_bindings _graphics_bindings{};
    _graphics_bindings.vertex_buffers[0] = state.graphics.vbuf;
    _graphics_bindings.index_buffer = state.graphics.ibuf;
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

void input(const sapp_event* event) {}

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
