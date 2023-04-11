
struct CameraUniform {
    vp: mat4x4<f32>,
};

struct D2SizeCameraUniform {
    view_size: vec2<f32>,
}

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOutput {
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @builtin(position) position: vec4<f32>,
}

struct MaterialUniform {
    color: vec4<f32>,
    alpha_test: f32,
}

struct Object {
    model: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera_uniform: CameraUniform;
@group(1) @binding(0) var<uniform> material_uniform: MaterialUniform;
@group(1) @binding(1) var texture_color: texture_2d<f32>;
@group(1) @binding(2) var sampler0: sampler;
var<push_constant> object: Object;

fn color_offset(va: f32) -> f32 {
    var val = va;
    if val > 0.8 {
        val = 1.0;
    } else if val > 0.6 {
        val = 0.8;
    } else {

    }
    return val;
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput{
    var output: VertexOutput;
    output.position = camera_uniform.vp * object.model * input.position;
    output.color = material_uniform.color;
    var val = 0.9;
    let times: i32 = 4;
    for (var i: i32 = 0; i < times; i++) {
        val *= input.color.a * color_offset(val);
    }

    output.color *= input.color;

    while false {
        output.color = vec4(0, 0, 0, 0);
    }

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32>{
    var color = input.color;
    color *= textureSample(texture_color, sampler0, input.uv);
    loop {
        color.a -= 0.1;
        break;
    }

    if color.a < material_uniform.alpha_test {
        discard;
    }
    return color;
}
