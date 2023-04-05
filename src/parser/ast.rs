use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::process::id;

use enums_arena::EnumsIdArena;

use super::token::AttributeType;
use super::token::Literal;
use super::token::SynToken;

pub trait Visitor<T> {
    fn visit(&self, t: &T) -> bool;
}

pub trait Node<'a> {
    fn next<V: Visitor<T>, T>(&self, visitor: V) -> bool;
}

#[derive(Eq, Hash, Clone, Copy)]
pub struct ExprId(ExpressionId<u32, ()>);
impl From<ExpressionId<u32, ()>> for ExprId {
    fn from(value: ExpressionId<u32, ()>) -> Self {
        Self(value)
    }
}
impl Into<ExpressionId<u32, ()>> for ExprId {
    fn into(self) -> ExpressionId<u32, ()> {
        self.0
    }
}

impl ExprId {
    pub fn ty(&self) -> ExpressionExtendEnum {
        self.0 .0
    }
}

impl Debug for ExprId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        get_expr(self.clone()).fmt(f)
    }
}

impl<'a> PartialEq for ExprId {
    fn eq(&self, other: &Self) -> bool {
        // if self.0.0 == ExpressionExtendEnum::Placement {
        //     if other.0.0 == ExpressionExtendEnum::Placement {
        //         return true
        //     }
        // }
        get_expr(self.clone()) == get_expr(other.clone())
    }
}

#[derive(Debug, Clone)]
pub struct PartialExprId {
    pub top: ExprId,
    pub partial: ExprId,
}

impl PartialExprId {
    pub fn is_empty(&self) -> bool {
        self.top == placement_expr_id()
    }

    pub fn new_empty() -> Self {
        Self {
            top: placement_expr_id(),
            partial:placement_expr_id(),
        }
    }
}


pub trait Expr<'a>: Clone {
    fn enum_expr(self) -> Expression<'a>;
    fn extract<'b>(_: &'b Expression<'a>) -> Option<&'b Self>
    where
        Self: Sized,
    {
        None
    }
}
macro_rules! use_expr_fn {
    ($t: tt) => {
        impl<'a> Expr<'a> for $t<'a> {
            fn enum_expr(self) -> Expression<'a> {
                Expression::$t(self)
            }
            fn extract<'b>(e: &'b Expression<'a>) -> Option<&'b Self>
            where
                Self: Sized,
            {
                match e {
                    Expression::$t(u) => Some(u),
                    _ => None,
                }
            }
        }
    };
}

#[derive(EnumsIdArena, Debug, PartialEq, Eq, Clone)]
pub enum Expression<'a> {
    UnaryExpression(UnaryExpression<'a>),
    BinaryExpression(BinaryExpression<'a>),
    FunctionCallExpression(FunctionCallExpression<'a>),
    PostfixExpression(PostfixExpression<'a>),
    IdentExpression(IdentExpression<'a>),
    LiteralExpression(LiteralExpression<'a>),
    TypeExpression(TypeExpression<'a>),
    ConcatExpression(ConcatExpression<'a>),
    ListExpression(ListExpression<'a>),
    ParenExpression(ParenExpression<'a>),
    Placement,
}

impl<'a> Expr<'a> for Expression<'a> {
    fn enum_expr(self) -> Expression<'a> {
        self
    }
}
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct UnaryExpression<'a> {
    pub op: SynToken,
    pub left: ExprId,
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(UnaryExpression);

impl<'a> UnaryExpression<'a> {
    pub fn new<E: Into<ExprId>>(op: SynToken, expr: E) -> Self {
        Self {
            op,
            left: expr.into(),
            _pd: PhantomData::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BinaryExpression<'a> {
    pub left: ExprId,
    pub op: SynToken,
    pub right: ExprId,
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(BinaryExpression);

impl<'a> BinaryExpression<'a> {
    pub fn new<E: Into<ExprId>, F: Into<ExprId>>(left: E, op: SynToken, right: F) -> Self {
        Self {
            left: left.into(),
            op,
            right: right.into(),
            _pd: PhantomData::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionCallExpression<'a> {
    pub ident: ExprId,        // ident
    pub args: Option<ExprId>, // list of expr
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(FunctionCallExpression);

impl<'a> FunctionCallExpression<'a> {
    pub fn new<E: Into<ExprId>>(ident: E, args: impl DoubleEndedIterator<Item = ExprId>) -> Self {
        let concat_exprs = args.map(|v| ConcatExpression::new_end(v));
        let mut prev: Option<ExprId> = None;
        for mut e in concat_exprs.rev() {
            e.right = prev;
            prev = Some(e.into());
        }

        Self {
            ident: ident.into(),
            args: prev,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_slice(ident: ExprId, args: &[ExprId]) -> Self {
        Self::new(ident, args.into_iter().cloned())
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum MemberOrExpr<'a> {
    Ident(&'a str),
    Index(ExprId),
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct PostfixExpression<'a> {
    pub ident: ExprId,
    pub postfix: MemberOrExpr<'a>,
}

use_expr_fn!(PostfixExpression);

impl<'a> PostfixExpression<'a> {
    pub fn new_ident(ident: ExprId, postfix: &'a str) -> Self {
        Self {
            ident,
            postfix: MemberOrExpr::Ident(postfix),
        }
    }
    pub fn new_index(ident: ExprId, postfix: ExprId) -> Self {
        Self {
            ident,
            postfix: MemberOrExpr::Index(postfix),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IdentExpression<'a> {
    pub name: &'a str,
    pub template_post_ident: Option<ExprId>, // list of type <,>
}

use_expr_fn!(IdentExpression);

impl<'a> IdentExpression<'a> {
    pub fn new(name: &'a str, template: Option<ExprId>) -> Self {
        Self {
            name,
            template_post_ident: template,
        }
    }
    pub fn new_ident(name: &'a str) -> Self {
        Self {
            name,
            template_post_ident: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LiteralExpression<'a> {
    pub literal: Literal,
    pub literal_str: &'a str,
}

impl<'a> PartialEq for LiteralExpression<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.literal_str == other.literal_str
    }
}

impl<'a> Eq for LiteralExpression<'a>{}

use_expr_fn!(LiteralExpression);

impl<'a> LiteralExpression<'a> {
    pub fn new(literal: Literal, literal_str: &'a str) -> Self {
        Self {
            literal,
            literal_str,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct TypeExpression<'a> {
    ty: Ty<'a>,
}

impl<'a> TypeExpression<'a> {
    pub fn new(ty: Ty<'a>) -> Self {
        Self { ty }
    }
}

use_expr_fn!(TypeExpression);

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ConcatExpression<'a> {
    pub cur: ExprId,
    pub right: Option<ExprId>,
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(ConcatExpression);

impl<'a> ConcatExpression<'a> {
    pub fn new(cur: ExprId, right: Option<ExprId>) -> Self {
        Self {
            cur,
            right,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_end(cur: ExprId) -> Self {
        Self {
            cur,
            right: None,
            _pd: PhantomData::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ListExpression<'a> {
    pub inner: Option<ExprId>,
    pub sep: SynToken,
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(ListExpression);

impl<'a> ListExpression<'a> {
    pub fn new_comma(inner: Option<ExprId>) -> Self {
        Self {
            inner,
            sep: SynToken::Comma,
            _pd: PhantomData::default(),
        }
    }
    pub fn new(inner: Option<ExprId>, sep: SynToken) -> Self {
        Self {
            inner,
            sep,
            _pd: PhantomData::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ParenExpression<'a> {
    pub inner: Option<ExprId>,
    pub l: SynToken,
    pub r: SynToken,
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(ParenExpression);

impl<'a> ParenExpression<'a> {
    pub fn new_paren(inner: Option<ExprId>) -> Self {
        Self {
            inner,
            l: SynToken::ParenLeft,
            r: SynToken::ParenRight,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_bracket(inner: Option<ExprId>) -> Self {
        Self {
            inner,
            l: SynToken::BracketLeft,
            r: SynToken::BracketRight,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_brace(inner: Option<ExprId>) -> Self {
        Self {
            inner,
            l: SynToken::BraceLeft,
            r: SynToken::BraceRight,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_angle(inner: Option<ExprId>) -> Self {
        Self {
            inner,
            l: SynToken::LessThan,
            r: SynToken::GreaterThan,
            _pd: PhantomData::default(),
        }
    }
}

// expr end

#[derive(Debug, PartialEq, Eq, Default)]
pub struct OptionallyTypedIdent<'a> {
    pub name: &'a str,
    pub ty: Option<Ty<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct GlobalVariableDecl<'a> {
    pub template_list: Option<ExprId>, // list of type
    pub ident: OptionallyTypedIdent<'a>,
    pub equals: Option<ExprId>,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct GlobalOverrideValueDecl<'a> {
    pub ident: OptionallyTypedIdent<'a>,
    pub equals: Option<ExprId>,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct GlobalConstValueDecl<'a> {
    pub ident: OptionallyTypedIdent<'a>,
    pub equals: ExprId,
}

#[derive(Debug, PartialEq, Eq)]
pub enum GlobalValueDecl<'a> {
    GlobalOverrideValueDecl(GlobalOverrideValueDecl<'a>),
    GlobalConstValueDecl(GlobalConstValueDecl<'a>),
}

#[derive(Debug, PartialEq, Eq)]
pub struct ConstAssertStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
    pub expr: ExprId,
}

// impl<'a> Node<'a> for ConstAssertStatement<'a> {
//     fn visit(&self, f: impl Fn(&Self) -> bool) -> bool {
//         let finish = f(self);
//         finish
//     }
// }

#[derive(Debug, PartialEq, Eq)]
pub struct CompoundStatement<'a> {
    pub attrs: Vec<Attribute<'a>>,
    pub statements: Vec<Statement<'a>>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Attribute<'a> {
    pub ty: AttributeType,
    pub expr: Option<ExprId>,
    pub diagnostic_control: Option<()>,
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Ty<'a> {
    Ident(&'a str),
    IdentTemplate((&'a str, ExprId)), // list of type
    None,
}

impl<'a> Ty<'a> {
    pub fn match_ident_template(&self, name: &str, expr: ExprId) -> bool {
        if let Self::IdentTemplate((n, e)) = self {
            if *n != name {
                return false;
            }
            // return expression_equal(e, &expr);
        }
        true
    }
}

impl<'a> Default for Ty<'a> {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct StructMember<'a> {
    pub ident: &'a str,
    pub ty: Ty<'a>,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct StructDecl<'a> {
    pub name: &'a str,
    pub members: Vec<StructMember<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct Param<'a> {
    pub name: &'a str,
    pub ty: Ty<'a>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct Statement<'a> {
    pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct FunctionDecl<'a> {
    pub name: &'a str,
    pub inputs: Vec<Param<'a>>,
    pub output: Option<(Vec<Attribute<'a>>, Ty<'a>)>,
    pub ast: Option<()>,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct TypeAliasDecl<'a> {
    pub name: &'a str,
    pub ty: Ty<'a>,
}

#[derive(Default)]
pub struct ParseResult<'a> {
    global_enables: Vec<&'a str>,
    decls: Vec<GlobalDecl<'a>>,
}

#[derive(Default)]
pub(crate) struct ParseContext<'a> {
    arena: ExpressionIdArena<'a, u32, ()>,
}

impl<'a> ParseContext<'a> {
    pub fn arena_mut(&mut self) -> &mut ExpressionIdArena<'a, u32, ()> {
        &mut self.arena
    }
    pub fn arena(&self) -> &ExpressionIdArena<'a, u32, ()> {
        &self.arena
    }
    pub fn reset(&mut self) {
        self.arena.clear();
    }
}

pub enum GlobalDecl<'a> {
    GlobalVariableDecl(GlobalVariableDecl<'a>),
    GlobalValueDecl(GlobalValueDecl<'a>),
    TypeAliasDecl(TypeAliasDecl<'a>),
    StructDecl(StructDecl<'a>),
    FunctionDecl(FunctionDecl<'a>),
    ConstAssertStatement(ConstAssertStatement<'a>),
    None,
}

thread_local! {
    pub(crate) static CTX: (RefCell<ParseContext<'static>>, ExprId) =  {
        let mut c = ParseContext::default();
        let id = c.arena_mut().alloc_placement();
        (RefCell::new(c), id.into())
    };
}

pub fn placement_expr_id() -> ExprId {
    CTX.with(|ctx| ctx.1)
}

// impl<'a> From<Expression<'a>> for ExprId {
//     fn from(value: Expression<'a>) -> Self {
//         CTX.with(move |ctx| {
//             let mut c = ctx.borrow_mut();
//             ExprId(c.arena_mut().alloc(value))
//         })
//     }
// }

pub fn get_expr<'a>(expr_id: ExprId) -> Expression<'a> {
    CTX.with(|ctx| unsafe {
        std::mem::transmute(ctx.0.borrow().arena().get(expr_id.into()).unwrap())
    })
}

impl<'a, E> From<E> for ExprId
where
    E: Expr<'a>,
{
    fn from(value: E) -> Self {
        CTX.with(move |ctx| {
            let mut c = ctx.0.borrow_mut();
            unsafe { ExprId(c.arena_mut().alloc(std::mem::transmute(value.enum_expr()))) }
        })
    }
}



// impl<'a, E> Node<'a> for E where E:Expr<'a> {
//     fn visit<T:E>(&self, f: impl Fn(&T) -> bool) -> bool {
//         let finish = f(self);
//         finish
//     }
// }

// // pub fn alloc_expr<'a, E>() where E: Expr<'a> {
// // }

// pub struct NodeVisitor {

// }

// impl Visitor for NodeVisitor {
//     fn visit<T>(&self, t: &T) -> bool {
//     }
// }

pub fn update_expr<F: for<'a> FnOnce(&mut Expression<'a>) -> Option<()>>(
    expr_id: ExprId,
    f: F,
) -> Option<()> {
    CTX.with(|ctx| {
        let mut expr = get_expr(expr_id);
        f(&mut expr)?;
        let mut c = ctx.0.borrow_mut();
        c.arena_mut().update(expr_id.into(), expr)?;
        Some(())
    })
}

pub fn update_expr_for1<'a, E: Expr<'a>, F: FnOnce(&mut E) -> Option<()>>(
    expr_id: ExprId,
    f: F,
) -> Option<()> {
    CTX.with(|ctx| {
        let expr = get_expr(expr_id);
        let mut val = E::extract(&expr)?.clone();
        f(&mut val)?;
        let val = unsafe {std::mem::transmute(val.enum_expr())};

        let mut c = ctx.0.borrow_mut();
        c.arena_mut().update(expr_id.into(), val)?;
        Some(())
    })
}

pub fn update_expr_for<'a, E: Expr<'a>, F: FnOnce(&mut E)>(
    expr_id: ExprId,
    f: F,
) -> Option<()> {
    CTX.with(|ctx| {
        let expr = get_expr(expr_id);
        let mut val = E::extract(&expr)?.clone();
        f(&mut val);
        let val = unsafe {std::mem::transmute(val.enum_expr())};

        let mut c = ctx.0.borrow_mut();
        c.arena_mut().update(expr_id.into(), val)?;
        Some(())
    })
}
