use strum::*;

#[derive(Debug, PartialEq, Clone, Eq)]
pub enum Integer {
    I(i32),
    U(u32),
    Abstract(i64),
}

impl From<Integer> for Literal {
    fn from(value: Integer) -> Self {
        Literal::Integer(value)
    }
}
impl From<bool> for Literal {
    fn from(value: bool) -> Self {
        Literal::Bool(value)
    }
}
impl From<Float> for Literal {
    fn from(value: Float) -> Self {
        Literal::Float(value)
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Float {
    F32(f32),
    F16(f32),
    Abstract(f64),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Literal {
    Bool(bool),
    Integer(Integer),
    Float(Float),
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString, Eq, PartialEq, Hash)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum Keyword {
    // type define
    Array = 0,
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
    #[strum(serialize = "texture_1d")]
    Texture1d,
    #[strum(serialize = "texture_2d")]
    Texture2d,
    #[strum(serialize = "texture_2d_array")]
    Texture2dArray,
    #[strum(serialize = "texture_3d")]
    Texture3d,
    TextureCube,
    TextureCubeArray,
    #[strum(serialize = "texture_multisampled_2d")]
    TextureMultisampled2d,
    #[strum(serialize = "texture_storage_1d")]
    TextureStorage1d,
    #[strum(serialize = "texture_storage_2d")]
    TextureStorage2d,
    #[strum(serialize = "texture_storage_2d_array")]
    TextureStorage2dArray,
    #[strum(serialize = "texture_storage_3d")]
    TextureStorage3d,
    #[strum(serialize = "texture_depth_2d")]
    TextureDepth2d,
    #[strum(serialize = "texture_depth_2d_array")]
    TextureDepth2dArray,
    TextureDepthCube,
    TextureDepthCubeArray,
    #[strum(serialize = "texture_depth_multisampled_2d")]
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
}

impl Keyword {
    pub fn is_type_keyword(&self) -> bool {
        (*self as u32) <= (Self::Vec4) as u32
    }
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString, Eq, PartialEq, Hash)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum ReservedWord {
    #[strum(serialize = "CompileShader")]
    CompileShader,
    #[strum(serialize = "ComputeShader")]
    ComputeShader,
    #[strum(serialize = "DomainShader")]
    DomainShader,
    #[strum(serialize = "GeometryShader")]
    GeometryShader,
    #[strum(serialize = "HullShader")]
    Hullshader,
    #[strum(serialize = "NULL")]
    #[warn(clippy::upper_case_acronyms)]
    NULL,
    #[strum(serialize = "Self")]
    SSelf,
    Abstract,
    Active,
    Alignas,
    Alignof,
    As,
    Asm,
    AsmFragment,
    Async,
    Attribute,
    Auto,
    Await,
    Become,
    Bf16,
    BindingArray,
    Cast,
    Catch,
    Class,
    CoAwait,
    CoReturn,
    CoYield,
    Coherent,
    ColumnMajor,
    Common,
    Compile,
    CompileFragment,
    Concept,
    ConstCast,
    Consteval,
    Constexpr,
    Constinit,
    Crate,
    Debugger,
    Decltype,
    Delete,
    Demote,
    DemoteToHelper,
    Do,
    DynamicCast,
    Enum,
    Explicit,
    Export,
    Extends,
    Extern,
    External,
    F64,
    Fallthrough,
    Filter,
    Final,
    Finally,
    Friend,
    From,
    Fxgroup,
    Get,
    Goto,
    Groupshared,
    Handle,
    Highp,
    I16,
    I64,
    I8,
    Impl,
    Implements,
    Import,
    Inline,
    Inout,
    Instanceof,
    Interface,
    Layout,
    Lowp,
    Macro,
    MacroRules,
    Match,
    Mediump,
    Meta,
    Mod,
    Module,
    Move,
    Mut,
    Mutable,
    Namespace,
    New,
    Nil,
    Noexcept,
    Noinline,
    Nointerpolation,
    Noperspective,
    Null,
    Nullptr,
    Of,
    Operator,
    Package,
    Packoffset,
    Partition,
    Pass,
    Patch,
    Pixelfragment,
    Precise,
    Precision,
    Premerge,
    Priv,
    Protected,
    Pub,
    Public,
    Quat,
    Readonly,
    Ref,
    Regardless,
    Register,
    ReinterpretCast,
    Requires,
    Resource,
    Restrict,
    #[strum(serialize = "self")]
    Sself,
    Set,
    Shared,
    Signed,
    Sizeof,
    Smooth,
    Snorm,
    Static,
    StaticAssert,
    StaticCast,
    Std,
    SubRoutine,
    Super,
    Target,
    Template,
    This,
    ThreadLocal,
    Throw,
    Trait,
    Try,
    Type,
    Typedef,
    Typeid,
    Typename,
    Typeof,
    U16,
    U64,
    U8,
    Union,
    Unless,
    Unorm,
    Unsafe,
    Unsized,
    Use,
    Using,
    Varying,
    Virtual,
    Volatile,
    Wgsl,
    Where,
    With,
    Writeonly,
    Yield,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum ContextName {
    Align,
    Binding,
    Builtin,
    Compute,
    Const,
    Fragment,
    Group,
    Id,
    Interpolate,
    Invariant,
    Location,
    Size,
    Vertex,
    WorkgroupSize,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum InterpolationType {
    Perspective,
    Linear,
    Flat,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum InterpolationSampling {
    Center,
    Centroid,
    Sample,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum BuiltinValue {
    VertexIndex,
    InstanceIndex,
    Position,
    FrontFacing,
    FragDepth,
    LocalInvocationId,
    LocalInvocationIndex,
    GlobalInvocationId,
    WorkgroupId,
    NumWorkgroups,
    SampleIndex,
    SampleMask,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum AccessMode {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum AddressSpace {
    Function,
    Private,
    Workgroup,
    Uniform,
    Storage,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum TexelFormat {
    Rgba8unorm,
    Rgba8snorm,
    Rgba8uint,
    Rgba8sint,
    Rgba16uint,
    Rgba16sint,
    Rgba16float,
    R32uint,
    R32sint,
    R32float,
    Rg32uint,
    Rg32sint,
    Rg32float,
    Rgba32uint,
    Rgba32sint,
    Rgba32float,
    Bgra8unorm,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum Extension {
    F16,
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString, PartialEq, Eq)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum SynToken {
    #[strum(serialize = "&")]
    And,
    #[strum(serialize = "&&")]
    AndAnd,
    #[strum(serialize = "->")]
    Arrow,
    #[strum(serialize = "@")]
    Attr,
    #[strum(serialize = "/")]
    Slash,
    #[strum(serialize = "!")]
    Bang,
    #[strum(serialize = "[")]
    BracketLeft,
    #[strum(serialize = "]")]
    BracketRight,
    #[strum(serialize = "{")]
    BraceLeft,
    #[strum(serialize = "}")]
    BraceRight,
    #[strum(serialize = ":")]
    Colon,
    #[strum(serialize = ",")]
    Comma,
    #[strum(serialize = "=")]
    Equal,
    #[strum(serialize = "==")]
    EqualEqual,
    #[strum(serialize = "!=")]
    NotEqual,
    #[strum(serialize = ">")]
    GreaterThan,
    #[strum(serialize = ">=")]
    GreaterThanEqual,
    #[strum(serialize = ">>")]
    ShiftRight,
    #[strum(serialize = "<")]
    LessThan,
    #[strum(serialize = "<=")]
    LessThanEqual,
    #[strum(serialize = "<<")]
    ShiftLeft,
    #[strum(serialize = "%")]
    Modulo,
    #[strum(serialize = "-")]
    Minus,
    #[strum(serialize = "--")]
    MinusMinus,
    #[strum(serialize = ".")]
    Period,
    #[strum(serialize = "+")]
    Plus,
    #[strum(serialize = "++")]
    PlusPlus,
    #[strum(serialize = "|")]
    Or,
    #[strum(serialize = "||")]
    OrOr,
    #[strum(serialize = "(")]
    ParenLeft,
    #[strum(serialize = ")")]
    ParenRight,
    #[strum(serialize = ";")]
    Semicolon,
    #[strum(serialize = "*")]
    Star,
    #[strum(serialize = "~")]
    Tilde,
    #[strum(serialize = "_")]
    Underscore,
    #[strum(serialize = "^")]
    Xor,
    #[strum(serialize = "+=")]
    PlusEqual,
    #[strum(serialize = "-=")]
    MinusEqual,
    #[strum(serialize = "*=")]
    TimesEqual,
    #[strum(serialize = "/=")]
    DivisionEqual,
    #[strum(serialize = "%=")]
    ModuloEqual,
    #[strum(serialize = "&=")]
    AndEqual,
    #[strum(serialize = "|=")]
    OrEqual,
    #[strum(serialize = "^=")]
    XorEqual,
    #[strum(serialize = ">==")]
    ShiftRightEqual,
    #[strum(serialize = "<==")]
    ShiftLeftEqual,
}

impl SynToken {
    pub fn max_chars() -> usize {
        3
    }
}

#[derive(Debug, Clone, Copy, Display, EnumIter, EnumString, PartialEq, Eq)]
#[repr(u32)]
#[strum(serialize_all = "snake_case", use_phf)]
pub enum AttributeType {
    Align,
    Binding,
    Builtin,
    Const,
    Diagnostic,
    Group,
    Id,
    Interpolate,
    Invariant,
    Location,
    MustUse,
    Size,
    WorkgroupSize,
    Vertex,
    Fragment,
    Compute,
}

impl AttributeType {
    pub fn min_chars() -> usize {
        2
    }
    pub fn max_chars() -> usize {
        13
    }

    pub fn extendable(&self) -> Option<(usize, usize)> {
        Some(match self {
            AttributeType::Align => (1, 1),
            AttributeType::Binding => (1, 1),
            AttributeType::Builtin => (1, 1),
            AttributeType::Diagnostic => (0, 0),
            AttributeType::Group => (1, 1),
            AttributeType::Id => (1, 1),
            AttributeType::Interpolate => (1, 2),
            AttributeType::Location => (1, 1),
            AttributeType::Size => (1, 1),
            AttributeType::WorkgroupSize => (1, 3),
            _ => return None,
        })
    }
}

pub enum TokenTy {
    Literal(Literal),
    Keyword(Keyword),
    ReservedWord(ReservedWord),
    SyntacticToken(SynToken),
    Identifier,
    ContextName(ContextName),
}

pub struct Token<'a> {
    str: &'a str,
    ty: TokenTy,
}
