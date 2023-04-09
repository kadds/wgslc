use super::token::AttributeType;
use super::token::Literal;
use super::token::SynToken;
use enums_arena::EnumsIdArena;
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use superslice::Ext;

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
            partial: placement_expr_id(),
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
    ($t: tt, $tt: tt) => {
        impl<'a> Expr<'a> for $tt<'a> {
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
    Unary(UnaryExpression<'a>),
    Binary(BinaryExpression<'a>),
    FunctionCall(FunctionCallExpression<'a>),
    Postfix(PostfixExpression<'a>),
    Ident(IdentExpression<'a>),
    Literal(LiteralExpression<'a>),
    Type(TypeExpression<'a>),
    Concat(ConcatExpression<'a>),
    List(ListExpression<'a>),
    Paren(ParenExpression<'a>),
    Placeholder,
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

use_expr_fn!(Unary, UnaryExpression);

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

use_expr_fn!(Binary, BinaryExpression);

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

use_expr_fn!(FunctionCall, FunctionCallExpression);

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

use_expr_fn!(Postfix, PostfixExpression);

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

use_expr_fn!(Ident, IdentExpression);

impl<'a> IdentExpression<'a> {
    pub fn new<E: Into<ExprId>>(name: &'a str, template: E) -> Self {
        Self {
            name,
            template_post_ident: Some(template.into()),
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

impl<'a> Eq for LiteralExpression<'a> {}

use_expr_fn!(Literal, LiteralExpression);

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

use_expr_fn!(Type, TypeExpression);

#[derive(PartialEq, Eq, Clone)]
pub struct ConcatExpression<'a> {
    pub cur: ExprId,
    pub right: Option<ExprId>,
    pub _pd: PhantomData<&'a ()>,
}

impl<'a> Debug for ConcatExpression<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.right {
            Some(mut cur) => {
                let mut list = f.debug_list();
                list.entry(&self.cur);
                loop {
                    let n = get_expr(cur);
                    if let Expression::Concat(e) = n {
                        list.entry(&e.cur);
                        if let Some(v) =  e.right {
                            cur = v;
                            continue;
                        }
                    } else {
                        list.entry(&n);
                    }
                    break;
                }
                list.finish()
            },
            _ => {
                self.cur.fmt(f)
            }
        }
    }
}

use_expr_fn!(Concat, ConcatExpression);

impl<'a> ConcatExpression<'a> {
    pub fn new<E: Into<ExprId>, F: Into<ExprId>>(cur: E, right: F) -> Self {
        Self {
            cur: cur.into(),
            right: Some(right.into()),
            _pd: PhantomData::default(),
        }
    }
    pub fn new_end<E: Into<ExprId>>(cur: E) -> Self {
        Self {
            cur: cur.into(),
            right: None,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_concat<E: Into<ExprId>>(mut list: impl Iterator<Item=E>) -> ExprId {
        let first = list.next().unwrap();
        if let Some(second_item) = list.next() {
            // second
            let p2: ExprId = Self::new_end(second_item).into();
            let p: ExprId = Self::new(first, p2).into();
            let mut c = p2;
            for item in list {
                let n: ExprId = Self::new_end(item).into();
                update_expr_for(c, |e: &mut ConcatExpression| {
                    e.right = Some(n);
                });
                c = n;
            }
            return p;
        } else {
            return first.into()
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ListExpression<'a> {
    pub inner: ExprId,
    pub sep: SynToken,
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(List, ListExpression);

impl<'a> ListExpression<'a> {
    pub fn new_comma<E: Into<ExprId>>(inner: E) -> Self {
        Self {
            inner: inner.into(),
            sep: SynToken::Comma,
            _pd: PhantomData::default(),
        }
    }
    pub fn new<E: Into<ExprId>>(inner: E, sep: SynToken) -> Self {
        Self {
            inner: inner.into(),
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

use_expr_fn!(Paren, ParenExpression);

impl<'a> ParenExpression<'a> {
    pub fn new_empty(l: SynToken) -> Self {
        let r = match l {
            SynToken::BracketLeft => SynToken::BracketRight,
            SynToken::BraceLeft => SynToken::BraceRight,
            SynToken::ParenLeft => SynToken::ParenRight,
            SynToken::LessThan => SynToken::GreaterThan,
            _ => panic!(""),
        };
        Self {
            inner: None,
            l,
            r,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_paren<E: Into<ExprId>>(inner: E) -> Self {
        Self {
            inner: Some(inner.into()),
            l: SynToken::ParenLeft,
            r: SynToken::ParenRight,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_bracket<E: Into<ExprId>>(inner: E) -> Self {
        Self {
            inner: Some(inner.into()),
            l: SynToken::BracketLeft,
            r: SynToken::BracketRight,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_brace<E: Into<ExprId>>(inner: E) -> Self {
        Self {
            inner: Some(inner.into()),
            l: SynToken::BraceLeft,
            r: SynToken::BraceRight,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_angle<E: Into<ExprId>>(inner: E) -> Self {
        Self {
            inner: Some(inner.into()),
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

// impl<'a> Node<'a> for ConstAssertStatement<'a> {
//     fn visit(&self, f: impl Fn(&Self) -> bool) -> bool {
//         let finish = f(self);
//         finish
//     }
// }

#[derive(Debug, EnumsIdArena, PartialEq, Eq)]
#[repr(u8)]
pub enum Statement<'a> {
    Block(BlockStatement<'a>),
    If(IfStatement<'a>),
    Compound(CompoundStatement<'a>),
    Assignment(AssignmentStatement<'a>),
    Switch(SwitchStatement<'a>),
    Loop(LoopStatement<'a>),
    For(ForStatement<'a>),
    While(WhileStatement<'a>),
    Break(BreakStatement<'a>),
    Continue(ContinueStatement<'a>),
    Return(ReturnStatement<'a>),
    FunctionCall(FunctionCallStatement<'a>),
    ConstAssert(ConstAssertStatement<'a>),
    Discard(DiscardStatement<'a>),

    Placeholder,
}

#[derive(Clone, Eq, Copy)]
pub struct StatmId(StatementId<u32, ()>);

impl From<StatementId<u32, ()>> for StatmId {
    fn from(value: StatementId<u32, ()>) -> Self {
        Self(value)
    }
}

impl Into<StatementId<u32, ()>> for StatmId {
    fn into(self) -> StatementId<u32, ()> {
        self.0
    }
}

impl Debug for StatmId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        get_statement(self.clone()).fmt(f)
    }
}

impl<'a> PartialEq for StatmId {
    fn eq(&self, other: &Self) -> bool {
        get_statement(self.clone()) == get_statement(other.clone())
    }
}

macro_rules! use_expr_new_for_statement {
    ($t: tt) => {
        impl<'a> $t<'a> {
            pub fn new<E: Into<ExprId>>(e: E) -> Self {
                Self {
                    expr: e.into(),
                    _pd: PhantomData::default(),
                }
            }
        }
    };
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CompoundStatement<'a> {
    pub attrs: Vec<Attribute<'a>>,
    pub statements: Vec<StatmId>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct AssignmentStatement<'a> {
    pub lhs: ExprId,
    pub rhs: ExprId,
    pub op: SynToken,
    pub _pd: PhantomData<&'a ()>,
}

impl<'a> AssignmentStatement<'a> {
    pub fn new<E: Into<ExprId>, F: Into<ExprId>>(lhs: E, rhs: F) -> Self {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
            op: SynToken::Equal,
            _pd: PhantomData::default(),
        }
    }
    pub fn new_op<E: Into<ExprId>, F: Into<ExprId>>(lhs: E, op: SynToken, rhs: F) -> Self {
        Self {
            lhs: lhs.into(),
            rhs: rhs.into(),
            op,
            _pd: PhantomData::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct IncrementStatement<'a> {
    pub lhs: ExprId,
    pub op: SynToken,
    pub _pd: PhantomData<&'a ()>,
}

impl<'a> IncrementStatement<'a> {
    pub fn new<E: Into<ExprId>, F: Into<ExprId>>(lhs: E, op: SynToken) -> Self {
        Self {
            lhs: lhs.into(),
            op,
            _pd: PhantomData::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BlockStatement<'a> {
    pub stmts: Vec<StatmId>,
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct IfStatement<'a> {
    pub cond: ExprId,
    pub accept: BlockStatement<'a>,
    pub reject: BlockStatement<'a>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct SwitchStatement<'a> {
    pub selector: ExprId,
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LoopStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ForStatement<'a> {
    pub init: StatmId,
    pub cond: StatmId,
    pub update: StatmId,
    pub body: StatmId, // block statement
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct WhileStatement<'a> {
    pub cond: StatmId,
    pub body: StatmId, // block statement
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BreakStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct BreakIfStatement<'a> {
    pub cond: ExprId,
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ContinueStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ContinuingStatement<'a> {
    pub cond: StatmId, // block
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ReturnStatement<'a> {
    pub expr: ExprId,
    pub _pd: PhantomData<&'a ()>,
}
use_expr_new_for_statement!(ReturnStatement);

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct FunctionCallStatement<'a> {
    pub expr: ExprId,
    pub _pd: PhantomData<&'a ()>,
}

use_expr_new_for_statement!(FunctionCallStatement);

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ConstAssertStatement<'a> {
    pub expr: ExprId,
    pub _pd: PhantomData<&'a ()>,
}
use_expr_new_for_statement!(ConstAssertStatement);

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub struct DiscardStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
}

// ---------

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Attribute<'a> {
    pub ty: AttributeType,
    pub exprs: Vec<ExprId>,
    pub diagnostic_control: Option<DiagnosticControl<'a>>,
}

#[derive(Debug, Clone)]
#[repr(u8)]
pub enum Ty<'a> {
    Ident(&'a str),
    IdentTemplate((&'a str, ExprId)), // list of type
    Literal((Literal, &'a str)),
    None,
}

impl<'a> PartialEq for Ty<'a> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ident(l0), Self::Ident(r0)) => l0 == r0,
            (Self::IdentTemplate(l0), Self::IdentTemplate(r0)) => l0 == r0,
            (Self::Literal(l0), Self::Literal(r0)) => l0.1 == r0.1,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}
impl<'a> Eq for Ty<'a> {}

impl<'a> Default for Ty<'a> {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct DiagnosticControl<'a> {
    pub name: &'a str,
    pub ident: &'a str,
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

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionDecl<'a> {
    pub name: &'a str,
    pub inputs: Vec<Param<'a>>,
    pub output: Option<(Vec<Attribute<'a>>, Ty<'a>)>,
    pub block: StatmId,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct TypeAliasDecl<'a> {
    pub name: &'a str,
    pub ty: Ty<'a>,
}

#[derive(Default)]
pub struct ParseResult<'a> {
    pub global_enables: Vec<&'a str>,
    pub decls: Vec<GlobalDecl<'a>>,
    ctx: ParseContext<'a>,
    // pub line_breaks: Vec<&'a str>,
}

impl<'a> ParseResult<'a> {
    pub(crate) fn finish(&mut self) {
        CTX.with(|ctx| {
            self.ctx = unsafe { std::mem::transmute(ctx.replace(ParseContext::default())) };
        })
    }
    // pub(crate) fn fill_line_breaks(&mut self, b: Vec<&'a str>) {
    //     self.line_breaks = b;
    // }
    // pub(crate) fn line_column(&self, str: &str) -> Option<Span> {
    //     if self.line_breaks.is_empty() {
    //         return None;
    //     }
    //     let b = self.line_breaks.lower_bound_by(|v| v.as_ptr().cmp(&str.as_ptr()));
    //     if b < self.line_breaks.len() {
    //     }
    // }
}

#[derive(Clone, Copy, Default)]
pub struct Span {
    pub beg: (usize, usize),
    pub end: (usize, usize),
}

impl Debug for Span {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Span")
            .field("start", &format!("{}:{}", self.beg.0, self.beg.1))
            .field("end", &format!("{}:{}", self.end.0, self.end.1))
            .finish()
    }
}

pub(crate) struct ParseContext<'a> {
    arena: ExpressionIdArena<'a, u32, ()>,
    statement_arena: StatementIdArena<'a, u32, ()>,
    pub placeholder: ExprId,
    pub statement_placeholder: StatmId,
    pub opt_level: u32,
}

impl<'a> Default for ParseContext<'a> {
    fn default() -> Self {
        let mut arena = ExpressionIdArena::default();
        let mut statement_arena = StatementIdArena::default();
        let placeholder = arena.alloc_placeholder().into();
        let statement_placeholder = statement_arena.alloc_placeholder().into();
        Self {
            arena,
            placeholder,
            opt_level: 0,
            statement_arena,
            statement_placeholder,
        }
    }
}

pub enum GlobalDecl<'a> {
    GlobalVariableDecl(GlobalVariableDecl<'a>),
    GlobalValueDecl(GlobalValueDecl<'a>),
    TypeAliasDecl(TypeAliasDecl<'a>),
    StructDecl(StructDecl<'a>),
    FunctionDecl(FunctionDecl<'a>),
    GlobalConstAssertStatement(ConstAssertStatement<'a>),
    None,
}

thread_local! {
    pub(crate) static CTX: RefCell<ParseContext<'static>> = RefCell::new(ParseContext::default());
}

pub fn placement_expr_id() -> ExprId {
    CTX.with(|ctx| ctx.borrow().placeholder)
}

pub fn placement_statm_id() -> StatmId {
    CTX.with(|ctx| ctx.borrow().statement_placeholder)
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
    CTX.with(|ctx| unsafe { std::mem::transmute(ctx.borrow().arena.get(expr_id.into()).unwrap()) })
}

pub fn get_statement<'a>(statement_id: StatmId) -> Statement<'a> {
    CTX.with(|ctx| unsafe {
        std::mem::transmute(
            ctx.borrow()
                .statement_arena
                .get(statement_id.into())
                .unwrap(),
        )
    })
}

impl<'a, E> From<E> for ExprId
where
    E: Expr<'a>,
{
    fn from(value: E) -> Self {
        CTX.with(move |ctx| {
            let mut c = ctx.borrow_mut();
            unsafe { ExprId(c.arena.alloc(std::mem::transmute(value.enum_expr()))) }
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
        let mut c = ctx.borrow_mut();
        c.arena.update(expr_id.into(), expr)?;
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
        let val = unsafe { std::mem::transmute(val.enum_expr()) };

        let mut c = ctx.borrow_mut();
        c.arena.update(expr_id.into(), val)?;
        Some(())
    })
}

pub fn update_expr_for<'a, E: Expr<'a>, F: FnOnce(&mut E)>(expr_id: ExprId, f: F) -> Option<()> {
    CTX.with(|ctx| {
        let expr = get_expr(expr_id);
        let mut val = E::extract(&expr)?.clone();
        f(&mut val);
        let val = unsafe { std::mem::transmute(val.enum_expr()) };

        let mut c = ctx.borrow_mut();
        c.arena.update(expr_id.into(), val)?;
        Some(())
    })
}
