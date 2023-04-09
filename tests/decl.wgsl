struct small_stride {
  a: array<f32,8> // stride 4
}
// Invalid, stride must be a multiple of 16
@group(0) @binding(0) var<uniform> invalid: small_stride;

struct wrapped_f32 {
  @size(16) elem: f32
}
struct big_stride {
  a: array<wrapped_f32,8> // stride 16
}
@group(0) @binding(1) var<uniform> valid: big_stride;     // Valid

struct S {
  x: f32
}
struct Invalid {
  a: S,
  b: f32 // invalid: offset between a and b is 4 bytes, but must be at least 16
}
@group(0) @binding(0) var<uniform> invalid2: Invalid;

struct Valid {
  a: S,
  @align(16) b: f32 // valid: offset between a and b is 16 bytes
}
@group(0) @binding(1) var<uniform> valid2: Valid;

struct A {                                     //             align(8)  size(32)
    u: f32,                                    // offset(0)   align(4)  size(4)
    v: f32,                                    // offset(4)   align(4)  size(4)
    w: vec2<f32>,                              // offset(8)   align(8)  size(8)
    @size(16) x: f32                           // offset(16)  align(4)  size(16)
}

struct B {                                     //             align(16) size(208)
    a: vec2<f32>,                              // offset(0)   align(8)  size(8)
    // -- implicit member alignment padding -- // offset(8)             size(8)
    b: vec3<f32>,                              // offset(16)  align(16) size(12)
    c: f32,                                    // offset(28)  align(4)  size(4)
    d: f32,                                    // offset(32)  align(4)  size(4)
    // -- implicit member alignment padding -- // offset(36)            size(12)
    @align(16) e: A,                           // offset(48)  align(16) size(32)
    f: vec3<f32>,                              // offset(80)  align(16) size(12)
    // -- implicit member alignment padding -- // offset(92)            size(4)
    g: array<A, 3>,    // element stride 32       offset(96)  align(8)  size(96)
    h: i32                                     // offset(192) align(4)  size(4)
    // -- implicit struct size padding --      // offset(196)           size(12)
}

@group(0) @binding(0)
var<uniform> uniform_buffer: B;

struct C {                                     //             align(8)  size(24)
    u: f32,                                    // offset(0)   align(4)  size(4)
    v: f32,                                    // offset(4)   align(4)  size(4)
    w: vec2<f32>,                              // offset(8)   align(8)  size(8)
    x: f32                                     // offset(16)  align(4)  size(4)
    // -- implicit struct size padding --      // offset(20)            size(4)
}

struct D {                                     //             align(16) size(160)
    a: vec2<f32>,                              // offset(0)   align(8)  size(8)
    // -- implicit member alignment padding -- // offset(8)             size(8)
    b: vec3<f32>,                              // offset(16)  align(16) size(12)
    c: f32,                                    // offset(28)  align(4)  size(4)
    d: f32,                                    // offset(32)  align(4)  size(4)
    // -- implicit member alignment padding -- // offset(36)            size(4)
    e: C,                                      // offset(40)  align(8)  size(24)
    f: vec3<f32>,                              // offset(64)  align(16) size(12)
    // -- implicit member alignment padding -- // offset(76)            size(4)
    g: array<A, 3>,    // element stride 24       offset(80)  align(8)  size(72)
    h: i32                                     // offset(152) align(4)  size(4)
    // -- implicit struct size padding --      // offset(156)           size(4)
}

@group(0) @binding(0)
var<storage,read_write> storage_buffer: D;