#[derive(Debug, PartialEq)]
pub enum Integer {
    I(i32),
    U(u32),
    Abstract(i64),
}

#[derive(Debug, PartialEq)]
pub enum Float {
    F32(f32),
    F16(f32),
    Abstract(f64),
}

#[derive(Debug, PartialEq)]
pub enum Literal {
    Bool(bool),
    Integer(Integer),
    Float(Float),
}

pub enum Keyword {
    // type define
    Array,
    Atomic,
    Bool,
    F32,
    F16,
    I32,
    Mat2x2,
    Mat2x3,
    Mat2x4,
    Mat3x2,
    Mat3x3,
    Mat3x4,
    Mat4x2,
    Mat4x3,
    Mat4x4,
    Ptr,
    Sampler,
    SamplerComparison,
    Texture1d,
    Texture2d,
    Texture2dArray,
    Texture3d,
    TextureCube,
    TextureCubeArray,
    TextureStorage1d,
    TextureStorage2d,
    TextureStorage2dArray,
    TextureStorage3d,
    TextureDepth2d,
    TextureDepth2dArray,
    TextureDepthCube,
    TextureDepthCubeArray,
    TextureDepthMultisampled2d,
    U32,
    Vec2,
    Vec3,
    Vec4,

    // other
    Alias,
    Bitcast,
    Break,
    Case,
    Const,
    ConstAssert,
    Continue,
    Continuing,
    Default,
    Discard,
    Else,
    Enable,
    False,
    Fn,
    For,
    If,
    Let,
    Loop,
    Override,
    Return,
    Struct,
    Switch,
    True,
    Var,
    While,
    // reserved
    // CompileShader,
    // ComputeShader,
    // DomainShader,
    // GeometryShader,
    // Hullshader,
    // NULL,
    // SSelf,
    // Abstract,
    // Active,
    // Alignas,
    // Alignof,
    // As,
    // Asm,
    // AsmFragment,
    // Async,
    // Attribute,
    // Auto,
    // Await,
    // Become,
    // Bf16,
    // BindingArray,
    // Cast,
}

pub enum SyntacticToken {}

pub enum ContextName {}

pub enum Ident<'a> {
    Ident(&'a str),
    MemberIdent(&'a str),
}

pub enum Token<'a> {
    Literal(Literal),
    Keyword(Keyword),
    ReservedWord(&'a str),
    SyntacticToken(SyntacticToken),
    Identifier(&'a str),
    ContextName(ContextName),
}

