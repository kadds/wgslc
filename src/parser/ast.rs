use super::token::AttributeType;
use serde::ser::{SerializeSeq, SerializeStruct};
use serde::{Serialize, Deserialize};
use super::token::Literal;
use super::token::SynToken;
use enums_arena::EnumsIdArena;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;

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
impl Serialize for ExprId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        get_expr(self.clone()).serialize(serializer)
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

#[derive(EnumsIdArena, Debug, PartialEq, Eq, Clone, Serialize)]
pub enum Expression<'a> {
    Unary(UnaryExpression<'a>),
    Binary(BinaryExpression<'a>),
    FunctionCall(FunctionCallExpression<'a>),
    Postfix(PostfixExpression<'a>),
    Ident(IdentExpression<'a>),
    Literal(LiteralExpression<'a>),
    Type(TypeExpression<'a>),
    List(ListExpression<'a>),
    Paren(ParenExpression<'a>),
    Placeholder,
}

impl<'a> Expr<'a> for Expression<'a> {
    fn enum_expr(self) -> Expression<'a> {
        self
    }
}
#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct FunctionCallExpression<'a> {
    pub ident: ExprId,        // ident
    pub args: Option<ExprId>, // list expr
    pub _pd: PhantomData<&'a ()>,
}

use_expr_fn!(FunctionCall, FunctionCallExpression);

impl<'a> FunctionCallExpression<'a> {
    pub fn new<E: Into<ExprId>>(ident: E, args: impl DoubleEndedIterator<Item = ExprId>) -> Self {
        // let concat_exprs = args.map(|v| ConcatExpression::new_end(v));
        let args = ListExpression::new_comma(args);

        Self {
            ident: ident.into(),
            args: Some(args.into()),
            _pd: PhantomData::default(),
        }
    }
    pub fn new_slice(ident: ExprId, args: &[ExprId]) -> Self {
        Self::new(ident, args.into_iter().cloned())
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub enum MemberOrExpr<'a> {
    Ident(&'a str),
    Index(ExprId),
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
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

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct TypeExpression<'a> {
    ty: Ty<'a>,
}

impl<'a> TypeExpression<'a> {
    pub fn new(ty: Ty<'a>) -> Self {
        Self { ty }
    }
}

use_expr_fn!(Type, TypeExpression);

#[derive(Eq, Clone)]
pub struct ListExpression<'a> {
    pub range: Range<u32>,
    pub sep: Option<SynToken>,
    pub _pd: PhantomData<&'a ()>,
}

impl<'a> PartialEq for ListExpression<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.range.clone().len() == other.range.clone().len() {
            for (lhs, rhs) in get_range_of_exprs(self.range.clone())
                .iter()
                .zip(get_range_of_exprs(other.range.clone()))
            {
                if !(*lhs).eq(rhs) {
                    return false;
                }
            }
            return true;
        }
        false
    }
}

impl<'a> Debug for ListExpression<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ret = get_range_of_exprs(self.range.clone());
        let ret: Vec<ExprId> = ret.iter().map(|v| v.clone()).collect();

        f.debug_struct("ListExpression")
            .field("list", &ret)
            .field("sep", &self.sep)
            .finish()
    }
}

impl<'a> Serialize for ListExpression<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        let ret = get_range_of_exprs(self.range.clone());
        let ret: Vec<ExprId> = ret.iter().map(|v| v.clone()).collect();
        let mut u = serializer.serialize_seq(None)?;
        for a in ret {
            u.serialize_element(&a)?;
        }
        u.end()
    }
}

use_expr_fn!(List, ListExpression);

impl<'a> ListExpression<'a> {
    pub fn new_comma<E: Into<ExprId>>(list: impl Iterator<Item = E>) -> Self {
        let iter = list.map(|v| v.into());
        Self {
            range: alloc_range_of_exprs(iter),
            sep: Some(SynToken::Comma),
            _pd: PhantomData::default(),
        }
    }

    pub fn new_none(list: impl Iterator<Item = ExprId>) -> Self {
        Self {
            range: alloc_range_of_exprs(list),
            sep: None,
            _pd: PhantomData::default(),
        }
    }
    pub fn new(range: Range<u32>, sep: SynToken) -> Self {
        Self {
            range,
            sep: Some(sep),
            _pd: PhantomData::default(),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
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

#[derive(Debug, PartialEq, Eq, Default, Clone, Serialize)]
pub struct OptionallyTypedIdent<'a> {
    pub name: &'a str,
    pub ty: Option<Ty<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default, Serialize)]
pub struct GlobalVariableDecl<'a> {
    pub template_list: Option<ExprId>, // list of type
    pub ident: OptionallyTypedIdent<'a>,
    pub equals: Option<ExprId>,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default, Serialize)]
pub struct GlobalOverrideValueDecl<'a> {
    pub ident: OptionallyTypedIdent<'a>,
    pub equals: Option<ExprId>,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct GlobalConstValueDecl<'a> {
    pub ident: OptionallyTypedIdent<'a>,
    pub equals: ExprId,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum GlobalValueDecl<'a> {
    GlobalOverrideValueDecl(GlobalOverrideValueDecl<'a>),
    GlobalConstValueDecl(GlobalConstValueDecl<'a>),
}

pub trait Stmt<'a>: Clone {
    fn enum_stmt(self) -> Statement<'a>;
    fn extract<'b>(_: &'b Statement<'a>) -> Option<&'b Self>
    where
        Self: Sized,
    {
        None
    }
}
macro_rules! use_stmt_fn {
    ($t: tt, $tt: tt) => {
        impl<'a> Stmt<'a> for $tt<'a> {
            fn enum_stmt(self) -> Statement<'a> {
                Statement::$t(self)
            }
            fn extract<'b>(e: &'b Statement<'a>) -> Option<&'b Self>
            where
                Self: Sized,
            {
                match e {
                    Statement::$t(u) => Some(u),
                    _ => None,
                }
            }
        }
    };
}

#[derive(Debug, EnumsIdArena, PartialEq, Eq, Serialize)]
#[repr(u8)]
pub enum Statement<'a> {
    If(IfStatement<'a>),
    ElIf(ElIfStatement<'a>),
    Else(ElseStatement<'a>),
    Compound(CompoundStatement<'a>),
    Assignment(AssignmentStatement<'a>),
    Decl(DeclStatement<'a>),
    Switch(SwitchStatement<'a>),
    Case(CaseStatement<'a>),
    Loop(LoopStatement<'a>),
    For(ForStatement<'a>),
    While(WhileStatement<'a>),
    Break(BreakStatement<'a>),
    Continue(ContinueStatement<'a>),
    Continuing(ContinuingStatement<'a>),
    Return(ReturnStatement<'a>),
    FunctionCall(FunctionCallStatement<'a>),
    ConstAssert(ConstAssertStatement<'a>),
    Discard(DiscardStatement<'a>),
    List(ListStatement<'a>),
    Placeholder,
}

#[derive(Clone, Eq, Copy)]
pub struct StmtId(StatementId<u32, ()>);

impl From<StatementId<u32, ()>> for StmtId {
    fn from(value: StatementId<u32, ()>) -> Self {
        Self(value)
    }
}

impl Into<StatementId<u32, ()>> for StmtId {
    fn into(self) -> StatementId<u32, ()> {
        self.0
    }
}

impl Debug for StmtId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        get_statement(self.clone()).fmt(f)
    }
}

impl Serialize for StmtId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        get_statement(self.clone()).serialize(serializer)
    }
}

impl<'a> PartialEq for StmtId {
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct AssignmentStatement<'a> {
    pub lhs: ExprId,
    pub rhs: ExprId,
    pub op: SynToken,
    pub _pd: PhantomData<&'a ()>,
}

use_stmt_fn!(Assignment, AssignmentStatement);

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

#[derive(Debug, PartialEq, Eq, Serialize)]
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct CompoundStatement<'a> {
    pub stmts: StmtId, // list
    pub attrs: Vec<Attribute<'a>>,
}
use_stmt_fn!(Compound, CompoundStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct DeclStatement<'a> {
    pub decl_ty: &'a str,
    pub template_list: Option<ExprId>,
    pub ident: OptionallyTypedIdent<'a>,
    pub assignment: Option<ExprId>,
}

use_stmt_fn!(Decl, DeclStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct IfStatement<'a> {
    pub cond: ExprId,
    pub accept: StmtId,
    pub reject: Option<StmtId>,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(If, IfStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct ElIfStatement<'a> {
    pub cond: ExprId,
    pub accept: StmtId,
    pub reject: Option<StmtId>,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(ElIf, ElIfStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct ElseStatement<'a> {
    pub accept: StmtId,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(Else, ElseStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct SwitchStatement<'a> {
    pub selector: ExprId,
    pub attrs: Vec<Attribute<'a>>,
    pub cases: StmtId, // list of CaseStatement
    pub _pd: PhantomData<&'a ()>,
}

use_stmt_fn!(Switch, SwitchStatement);


#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct CaseStatement<'a> {
    pub selector: Option<ExprId>,
    pub body: StmtId,
    pub _pd: PhantomData<&'a ()>,
}

use_stmt_fn!(Case, CaseStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct LoopStatement<'a> {
    pub attrs: Vec<Attribute<'a>>,
    pub body: StmtId, // list statement
    pub continuing: Option<StmtId>,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(Loop, LoopStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct ForStatement<'a> {
    pub init: Option<StmtId>,
    pub cond: Option<ExprId>,
    pub update: Option<StmtId>,
    pub body: StmtId, // block statement
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(For, ForStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct WhileStatement<'a> {
    pub cond: ExprId,
    pub body: StmtId, // block statement
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(While, WhileStatement);

#[derive(Debug, PartialEq, Eq, Clone, Default, Serialize)]
pub struct BreakStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(Break, BreakStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct BreakIfStatement<'a> {
    pub cond: ExprId,
    pub _pd: PhantomData<&'a ()>,
}

#[derive(Debug, PartialEq, Eq, Clone, Default, Serialize)]
pub struct ContinueStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(Continue, ContinueStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct ContinuingStatement<'a> {
    pub attrs: Vec<Attribute<'a>>,
    pub body: StmtId, // list statement
    pub break_if: Option<ExprId>,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(Continuing, ContinuingStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct ReturnStatement<'a> {
    pub expr: ExprId,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(Return, ReturnStatement);
use_expr_new_for_statement!(ReturnStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct FunctionCallStatement<'a> {
    pub expr: ExprId,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(FunctionCall, FunctionCallStatement);

use_expr_new_for_statement!(FunctionCallStatement);

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct ConstAssertStatement<'a> {
    pub expr: ExprId,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(ConstAssert, ConstAssertStatement);
use_expr_new_for_statement!(ConstAssertStatement);

#[derive(Debug, PartialEq, Eq, Clone, Default, Serialize)]
pub struct DiscardStatement<'a> {
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(Discard, DiscardStatement);

#[derive(Eq, Clone)]
pub struct ListStatement<'a> {
    pub range: Range<u32>,
    pub _pd: PhantomData<&'a ()>,
}
use_stmt_fn!(List, ListStatement);
impl<'a> ListStatement<'a> {
    pub fn new<S: Into<StmtId>>(list: impl Iterator<Item = S>) -> Self {
        let iter = list.map(|v| v.into());
        Self {
            range: alloc_range_of_stmts(iter),
            _pd: PhantomData::default(),
        }
    }
}

impl<'a> PartialEq for ListStatement<'a> {
    fn eq(&self, other: &Self) -> bool {
        if self.range.clone().len() == other.range.clone().len() {
            for (lhs, rhs) in get_range_of_stmts(self.range.clone())
                .iter()
                .zip(get_range_of_stmts(other.range.clone()))
            {
                if !(*lhs).eq(rhs) {
                    return false;
                }
            }
            return true;
        }
        false
    }
}

impl<'a> Debug for ListStatement<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ret = get_range_of_stmts(self.range.clone());
        let ret: Vec<StmtId> = ret.iter().map(|v| v.clone()).collect();

        f.debug_struct("ListStatement").field("list", &ret).finish()
    }
}

impl<'a> Serialize for ListStatement<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        let ret = get_range_of_stmts(self.range.clone());
        let ret: Vec<StmtId> = ret.iter().map(|v| v.clone()).collect();
        let mut u = serializer.serialize_seq(None)?;
        for val in ret {
            u.serialize_element(&val)?;
        }
        u.end()
    }
}

// ---------

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct Attribute<'a> {
    pub ty: AttributeType,
    pub exprs: Vec<ExprId>,
    pub diagnostic_control: Option<DiagnosticControl<'a>>,
}

#[derive(Debug, Clone, Serialize)]
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

#[derive(Debug, PartialEq, Eq, Clone, Serialize)]
pub struct DiagnosticControl<'a> {
    pub name: &'a str,
    pub ident: &'a str,
}

#[derive(Debug, PartialEq, Eq, Default, Serialize)]
pub struct StructMember<'a> {
    pub ident: &'a str,
    pub ty: Ty<'a>,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default, Serialize)]
pub struct StructDecl<'a> {
    pub name: &'a str,
    pub members: Vec<StructMember<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default, Serialize)]
pub struct Param<'a> {
    pub name: &'a str,
    pub ty: Ty<'a>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct FunctionDecl<'a> {
    pub name: &'a str,
    pub inputs: Vec<Param<'a>>,
    pub output: Option<(Vec<Attribute<'a>>, Ty<'a>)>,
    pub block: StmtId,
    pub attrs: Vec<Attribute<'a>>,
}

#[derive(Debug, PartialEq, Eq, Default, Serialize)]
pub struct TypeAliasDecl<'a> {
    pub name: &'a str,
    pub ty: Ty<'a>,
}

#[derive(Default)]

pub struct ParseResult<'a> {
    pub global_enables: Vec<&'a str>,
    pub decls: Vec<GlobalDecl<'a>>,
    ctx: ParseContext<'a>,
}

impl<'a> Debug for ParseResult<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParseResult")
            .field("global_enables", &self.global_enables)
            .field("decls", &self.decls)
            .finish()
    }
}

impl<'a> Serialize for ParseResult<'a> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer {
        let mut u = serializer.serialize_struct("global_enables", 2)?;
        u.serialize_field("global_enables", &self.global_enables);
        u.serialize_field("decls", &self.decls);
        u.end()
    }
}

impl<'a> ParseResult<'a> {
    pub(crate) fn finish(&mut self) {
        CTX.with(|ctx| {
            self.ctx = unsafe { std::mem::transmute(ctx.replace(ParseContext::default())) };
        })
    }
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
    expr_vecs: Vec<ExprId>,
    statement_vecs: Vec<StmtId>,

    pub placeholder: ExprId,
    pub statement_placeholder: StmtId,
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
            expr_vecs: vec![],
            statement_vecs: vec![],
            opt_level: 0,
            statement_arena,
            statement_placeholder,
        }
    }
}

#[derive(Debug, Serialize)]
pub enum GlobalDecl<'a> {
    GlobalVariableDecl(GlobalVariableDecl<'a>),
    GlobalValueDecl(GlobalValueDecl<'a>),
    TypeAliasDecl(TypeAliasDecl<'a>),
    StructDecl(StructDecl<'a>),
    FunctionDecl(FunctionDecl<'a>),
    GlobalConstAssertStatement(StmtId),
    None,
}

thread_local! {
    pub(crate) static CTX: RefCell<ParseContext<'static>> = RefCell::new(ParseContext::default());
}

pub fn placement_expr_id() -> ExprId {
    CTX.with(|ctx| ctx.borrow().placeholder)
}

pub fn placement_stmt_id() -> StmtId {
    CTX.with(|ctx| ctx.borrow().statement_placeholder)
}

fn alloc_range_of_exprs(list: impl Iterator<Item = ExprId>) -> Range<u32> {
    CTX.with(|ctx| {
        let l = {
            let ctx = ctx.borrow_mut();
            ctx.expr_vecs.len() as u32
        };
        for item in list {
            let mut ctx = ctx.borrow_mut();
            ctx.expr_vecs.push(item);
        }
        let r = {
            let ctx = ctx.borrow_mut();
            ctx.expr_vecs.len() as u32
        };
        l..r
    })
}

fn get_range_of_exprs<'a>(range: Range<u32>) -> &'a [ExprId] {
    let range = range.start as usize..range.end as usize;
    CTX.with(|ctx| {
        let ctx = ctx.borrow();
        let iter = ctx.expr_vecs.iter();

        unsafe { std::mem::transmute(&ctx.expr_vecs[range]) }
    })
}

fn alloc_range_of_stmts(list: impl Iterator<Item = StmtId>) -> Range<u32> {
    CTX.with(|ctx| {
        let l = {
            let ctx = ctx.borrow_mut();
            ctx.statement_vecs.len() as u32
        };
        for item in list {
            let mut ctx = ctx.borrow_mut();
            ctx.statement_vecs.push(item);
        }
        let r = {
            let ctx = ctx.borrow_mut();
            ctx.statement_vecs.len() as u32
        };
        l..r
    })
}

fn get_range_of_stmts<'a>(range: Range<u32>) -> &'a [StmtId] {
    let range = range.start as usize..range.end as usize;
    CTX.with(|ctx| {
        let ctx = ctx.borrow();
        let iter = ctx.statement_vecs.iter();

        unsafe { std::mem::transmute(&ctx.statement_vecs[range]) }
    })
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

pub fn get_statement<'a>(statement_id: StmtId) -> Statement<'a> {
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

impl<'a, S> From<S> for StmtId
where
    S: Stmt<'a>,
{
    fn from(value: S) -> Self {
        CTX.with(move |ctx| {
            let mut c = ctx.borrow_mut();
            unsafe {
                StmtId(
                    c.statement_arena
                        .alloc(std::mem::transmute(value.enum_stmt())),
                )
            }
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

pub fn update_stmt_for<'a, S: Stmt<'a>, F: FnOnce(&mut S)>(stmt_id: StmtId, f: F) -> Option<()> {
    CTX.with(|ctx| {
        let expr = get_statement(stmt_id);
        let mut val = S::extract(&expr)?.clone();
        f(&mut val);
        let val = unsafe { std::mem::transmute(val.enum_stmt()) };

        let mut c = ctx.borrow_mut();
        c.statement_arena.update(stmt_id.into(), val)?;
        Some(())
    })
}
