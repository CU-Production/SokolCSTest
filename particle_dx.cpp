#define SOKOL_IMPL
#define SOKOL_NO_ENTRY
#define SOKOL_D3D11
#include "sokol_app.h"
#include "sokol_gfx.h"
#include "sokol_glue.h"
#include "sokol_log.h"

#include "HandmadeMath.h"

#include <random>
#include <vector>

constexpr uint32_t SCREEN_WIDTH = 800;
constexpr uint32_t SCREEN_HEIGHT = 600;

constexpr uint32_t PARTICLE_COUNT = 8192;

struct cs_params_t{
    float dt;
    int32_t num_particles;
};

struct particle_t{
    HMM_Vec2 pos;
    HMM_Vec2 vel;
    HMM_Vec4 color;
};

struct {
    struct {
        sg_buffer buf;
        sg_pipeline pip;
    } compute;
    struct {
        sg_pipeline pip;
        sg_pass_action pass_action;
    } graphics;
} state;

void init() {
    sg_desc _sg_desc{};
    _sg_desc.environment = sglue_environment();
    _sg_desc.logger.func = slog_func;
    sg_setup(&_sg_desc);

    // compute
    {
        std::default_random_engine rndEngine((uint32_t)time(nullptr));
        std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

        std::vector<particle_t> particles{PARTICLE_COUNT};
        for (uint32_t i = 0; i < PARTICLE_COUNT; i++) {
            float r = 0.25f * std::sqrt(rndDist(rndEngine));
            float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846f;
            float x = r * std::cos(theta) * SCREEN_HEIGHT / SCREEN_HEIGHT;
            float y = r * std::sin(theta);
            particles[i].pos = HMM_V2(x, y);
            particles[i].vel = HMM_Norm(HMM_V2(x, y)) * 0.25f;
            particles[i].color = HMM_V4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine));
        }

        sg_buffer_desc _sg_buffer_desc{};
        _sg_buffer_desc.usage.storage_buffer = true;
        _sg_buffer_desc.data.ptr = particles.data();
        _sg_buffer_desc.data.size = sizeof(particle_t) * PARTICLE_COUNT;
        _sg_buffer_desc.label = "particle-buffer";
        state.compute.buf = sg_make_buffer(&_sg_buffer_desc);

        sg_shader_desc _sg_compute_shader_desc{};
        _sg_compute_shader_desc.compute_func.source = R"(
cbuffer params: register(b0) {
  float dt;
  uint num_particles;
};

struct particle_t {
  float2 pos;
  float2 vel;
  float4 color;
};

RWStructuredBuffer<particle_t> prt: register(u0);

[numthreads(64,1,1)]
void main(uint3 dtid: SV_DispatchThreadID) {
  uint idx = dtid.x;;
  if (idx >= num_particles) {
    return;
  }

  float2 pos = prt[idx].pos;
  float2 vel = prt[idx].vel;
  pos = pos + vel * dt;

  // Flip movement at window border
  if ( (pos.x <= -1.0) || (pos.x >= 1.0) ) {
      vel.x *= -1.0;
  }
  if ( (pos.y <= -1.0) || (pos.y >= 1.0) ) {
      vel.y *= -1.0;
  }

  prt[idx].pos = pos;
  prt[idx].vel = vel;
}
)";

        _sg_compute_shader_desc.uniform_blocks[0].stage = SG_SHADERSTAGE_COMPUTE;
        _sg_compute_shader_desc.uniform_blocks[0].size = sizeof(cs_params_t);
        _sg_compute_shader_desc.uniform_blocks[0].hlsl_register_b_n = 0;

        _sg_compute_shader_desc.storage_buffers[0].stage = SG_SHADERSTAGE_COMPUTE;
        _sg_compute_shader_desc.storage_buffers[0].readonly = false;
        _sg_compute_shader_desc.storage_buffers[0].hlsl_register_u_n = 0;

        _sg_compute_shader_desc.label = "compute-shader";

        sg_shader compute_shd = sg_make_shader(&_sg_compute_shader_desc);

        sg_pipeline_desc _compute_pipeline_desc{};
        _compute_pipeline_desc.compute = true;
        _compute_pipeline_desc.shader = compute_shd;
        _compute_pipeline_desc.label = "compute-pipeline";

        state.compute.pip = sg_make_pipeline(&_compute_pipeline_desc);
    }

    // graphics
    {
        sg_shader_desc _shader_desc{};
        _shader_desc.vertex_func.source = R"(
struct particle_t {
  float2 pos;
  float2 vel;
  float4 color;
};

StructuredBuffer<particle_t> prt: register(t0);

struct vs_out {
  float4 pos: SV_Position;
  float4 color: COLOR0;
};

static const float2 vertices[4] = {
  {-1.0f, -1.0f,},
  { 1.0f, -1.0f,},
  {-1.0f,  1.0f,},
  { 1.0f,  1.0f,},
};

static const int indices[6] = { 0, 1, 2, 1, 3, 2 };

static const float point_size = 20.0f;
static const float2 screen_size = {800.0f, 600.0f};

vs_out main(uint gl_VertexID: SV_VertexID, uint gl_InstanceID: SV_InstanceID) {
  const float half_point_size = point_size * 0.5f;
  float2 position = half_point_size * vertices[indices[gl_VertexID]] / screen_size;

  vs_out outp = (vs_out)0;
  float2 pos = prt[gl_InstanceID].pos + position;
  float4 color = prt[gl_InstanceID].color;
  outp.pos = float4(pos, 0.0f, 1.0f);
  outp.color = color;
  return outp;
}
)";
        _shader_desc.fragment_func.source = R"(
struct vs_out {
  float4 pos: SV_Position;
  float4 color: COLOR0;
};

float4 main(vs_out outp): SV_Target0 {
  return outp.color;
}
)";
        _shader_desc.storage_buffers[0].stage = SG_SHADERSTAGE_VERTEX;
        _shader_desc.storage_buffers[0].readonly = true;
        _shader_desc.storage_buffers[0].hlsl_register_t_n = 0;

        _shader_desc.label = "fragment-shader";

        sg_shader graphics_shd = sg_make_shader(&_shader_desc);

        sg_pipeline_desc _pipeline_desc{};
        _pipeline_desc.shader = graphics_shd;
        _pipeline_desc.primitive_type = SG_PRIMITIVETYPE_TRIANGLES;
        state.graphics.pip = sg_make_pipeline(&_pipeline_desc);

        state.graphics.pass_action.colors[0] = { .load_action=SG_LOADACTION_CLEAR, .clear_value={0.2f, 0.3f, 0.3f, 1.0f } };
    }


 }

void frame() {
    const double dt = sapp_frame_duration();

    const cs_params_t cs_params = { (float)dt, PARTICLE_COUNT};

    // compute pass
    sg_bindings _compute_bindings{};
    _compute_bindings.storage_buffers[0] = state.compute.buf;
    sg_pass _compute_pass = { .compute=true, .label="compute_pass" };
    sg_begin_pass(&_compute_pass);
    sg_apply_pipeline(state.compute.pip);
    sg_apply_bindings(_compute_bindings);
    sg_apply_uniforms(0, SG_RANGE(cs_params));
    sg_dispatch((PARTICLE_COUNT + 63)/64, 1, 1);
    sg_end_pass();

    // graphics pass
    sg_bindings _graphics_bindings{};
    _graphics_bindings.storage_buffers[0] = state.compute.buf;
    sg_pass _graphics_pass = { .action=state.graphics.pass_action, .swapchain=sglue_swapchain() };
    sg_begin_pass(&_graphics_pass);
    sg_apply_pipeline(state.graphics.pip);
    sg_apply_bindings(_graphics_bindings);
    sg_draw(0, 6, PARTICLE_COUNT);
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
    desc.window_title = "sokol cs particle (DX)",
    desc.icon.sokol_default = true,
    desc.logger.func = slog_func;
    sapp_run(&desc);

    return 0;
}
