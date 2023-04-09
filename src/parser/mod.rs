use std::{fmt::Debug, marker::PhantomData, ops::RangeFrom, ptr::slice_from_raw_parts};

use nom::{
    error::{ContextError, FromExternalError, ParseError},
    sequence::terminated,
    IResult,
};

use self::{
    ast::Span,
    lex::{global_decls, global_directives, linebreak, next_linebreak},
};

mod ast;
mod lex;
mod number;
mod token;

pub fn incomplete_or_else<F, G, Input, Output, Error: nom::error::ParseError<Input>>(
    mut child: G,
    or_fn: F,
) -> impl FnMut(Input) -> IResult<Input, Output, Error>
where
    Input: nom::InputIter + nom::InputTake + Clone,
    F: Fn() -> Output,
    G: nom::Parser<Input, Output, Error>,
{
    move |i: Input| match child.parse(i.clone()) {
        Err(err) => match err {
            nom::Err::Incomplete(_n) => Ok((i, or_fn())),
            e => Err(e),
        },
        Ok(val) => Ok(val),
    }
}

pub fn one_of_or_else<F, Input, T, Error: nom::error::ParseError<Input>>(
    list: T,
    or_fn: F,
) -> impl Fn(Input) -> IResult<Input, char, Error>
where
    Input: nom::InputIter + nom::InputTake + nom::Slice<RangeFrom<usize>>,
    T: nom::FindToken<<Input as nom::InputIter>::Item>,
    <Input as nom::InputIter>::Item: nom::AsChar + Copy,
    F: Fn() -> char,
{
    use nom::AsChar;
    move |i: Input| match (i).iter_elements().next().map(|c| (c, list.find_token(c))) {
        Some((c, true)) => Ok((i.slice(c.len()..), c.as_char())),
        _ => Ok((i, or_fn())),
    }
}

pub fn separated_list1_ext_sep<I, O, O2, E, F, G>(
    mut sep: G,
    mut f: F,
) -> impl FnMut(I) -> IResult<I, Vec<O>, E>
where
    I: Clone + nom::InputLength,
    F: nom::Parser<I, O, E>,
    G: nom::Parser<I, O2, E>,
    E: nom::error::ParseError<I>,
{
    move |mut i: I| {
        let mut res = Vec::new();

        // Parse the first element
        match f.parse(i.clone()) {
            Err(e) => return Err(e),
            Ok((i1, o)) => {
                res.push(o);
                i = i1;
            }
        }

        loop {
            let len = i.input_len();
            match sep.parse(i.clone()) {
                Err(nom::Err::Error(_)) => return Ok((i, res)),
                Err(e) => return Err(e),
                Ok((i1, _)) => {
                    // infinite loop check: the parser must always consume
                    if i1.input_len() == len {
                        return Err(nom::Err::Error(E::from_error_kind(
                            i1,
                            nom::error::ErrorKind::SeparatedList,
                        )));
                    }

                    match f.parse(i1.clone()) {
                        Err(nom::Err::Failure(f)) => return Err(nom::Err::Failure(f)),
                        Err(_) => return Ok((i1, res)),
                        Ok((i2, o)) => {
                            res.push(o);
                            i = i2;
                        }
                    }
                }
            }
        }
    }
}

pub fn dbg<F, I, O, E>(mut f: F, context: &'static str) -> impl FnMut(I) -> IResult<I, O, E>
where
    F: FnMut(I) -> IResult<I, O, E>,
    I: Debug + Clone,
    E: Debug,
{
    move |i| match f(i.clone()) {
        Err(e) => {
            // let slice = substr_n(&format!("{:?}", e), 128 "...");
            let islice = substr_n(&format!("{:?}", i), 24, "...");
            log::error!("{}: find error # {:?} # at: {:?}", context, e, islice);
            Err(e)
        }
        a => a,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ErrorKind {
    #[error("ident")]
    ExpectIdent,
    #[error("literal")]
    ExpectLiteral,
    #[error("expect space")]
    ExpectSpace,
    #[error("external error")]
    External,
    #[error("parse literal")]
    ParseLiteral,

    #[error("line break")]
    ExpectLineBreak,

    #[error("expect keyword")]
    ExpectKeyword,
    #[error("expect attribute")]
    ExpectAttribute,

    #[error("expect template ident")]
    ExpectTemplateIdent,

    #[error("eof")]
    Eof,
}

pub(crate) struct ErrContext<'a> {
    msg: String,
    input: &'a str,
}

impl<'a> Debug for ErrContext<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input = substr_n(self.input, 24, "...");
        f.debug_struct("")
            .field("msg", &self.msg)
            .field("input", &input)
            .finish()
    }
}

pub struct Error<'a> {
    pub neal_by: &'a str,
    pub kind: ErrorKind,
    pub message: String,
    pub(crate) context: Vec<ErrContext<'a>>,
    pub span: Option<Span>,
}

impl<'a> Debug for Error<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = substr_n(self.neal_by, 32, "...");
        f.debug_struct("Error")
            .field("neal_by", &n)
            .field("kind", &self.kind)
            .field("message", &self.message)
            .field("context", &self.context)
            .field("at", &self.span)
            .finish()
    }
}

impl<'a> Error<'a> {
    pub fn new(s: &'a str, kind: ErrorKind) -> Self {
        Self {
            neal_by: s,
            kind,
            message: String::new(),
            context: Vec::new(),
            span: None,
        }
    }
}

impl<'a, I, E> FromExternalError<I, E> for Error<'a>
where
    I: Debug + Into<&'a str>,
    E: Debug,
{
    fn from_external_error(input: I, kind: nom::error::ErrorKind, e: E) -> Self {
        Self {
            neal_by: input.into(),
            kind: ErrorKind::External,
            message: format!("{:?} {:?}", e, kind),
            context: Vec::new(),
            span: None,
        }
    }
}

impl<'a, I> ParseError<I> for Error<'a>
where
    I: Debug + Into<&'a str>,
{
    fn from_error_kind(input: I, kind: nom::error::ErrorKind) -> Self {
        Self {
            neal_by: input.into(),
            kind: ErrorKind::External,
            message: format!("{:?}", kind),
            context: Vec::new(),
            span: None,
        }
    }

    fn append(input: I, kind: nom::error::ErrorKind, mut other: Self) -> Self {
        other.message = format!("{} inner ({:?})", other.message, kind);
        other
    }

    fn or(self, other: Self) -> Self {
        if other.context.len() < self.context.len() {
            self
        } else {
            other
        }
    }
}

impl<'a, I> ContextError<I> for Error<'a>
where
    I: Debug + Into<&'a str>,
{
    fn add_context(input: I, ctx: &'static str, mut other: Self) -> Self {
        other.context.push(ErrContext {
            msg: ctx.to_owned(),
            input: input.into(),
        });
        other
    }
}

pub struct Parser {
    line_info: bool,
}

impl Parser {
    pub fn new(line_info: bool) -> Self {
        Self { line_info }
    }
    fn get_line_info<'a>(&self, input: &'a str, target: &'a str) -> Span {
        if !self.line_info {
            return Span::default();
        }
        let mut i = input;
        let mut line = 1;
        while let Ok((i2, _)) = next_linebreak(i) {
            if i2.as_ptr() > target.as_ptr() {
                unsafe {
                    let len = target.as_ptr() as usize - i.as_ptr() as usize;
                    let slice = slice_from_raw_parts(i.as_ptr(), len);
                    let str = std::str::from_utf8(std::mem::transmute(slice)).unwrap();
                    return Span {
                        beg: (line, str.len() + 1),
                        end: (0, 0),
                    };
                }
            }
            i = i2;
            line += 1;
        }
        Span::default()
    }

    fn map_ret<'a>(&self, input: &'a str, e: nom::Err<Error<'a>>) -> Error<'a> {
        match e {
            nom::Err::Incomplete(e) => {
                panic!();
            }
            nom::Err::Error(e) => {
                panic!();
            }
            nom::Err::Failure(mut f) => {
                if self.line_info {
                    f.span = Some(self.get_line_info(input, f.neal_by))
                }
                f
            }
        }
    }

    pub fn parse<'a>(&self, input: &'a str) -> Result<ast::ParseResult<'a>, Error<'a>> {
        let mut ret = ast::ParseResult::default();
        // try global directive first
        let (input, output) = global_directives(input).map_err(|e| {
            ret.finish();
            self.map_ret(input, e)
        })?;
        ret.global_enables = output;

        let (input, output) = global_decls(input).map_err(|e| {
            ret.finish();
            self.map_ret(input, e)
        })?;
        ret.decls = output;
        ret.finish();

        // try global decl
        Ok(ret)
    }
}

fn substr_n(str: &str, n: usize, postfix: &str) -> String {
    let mut chars = str.chars();
    let mut c = 0;
    let mut chs = 0;
    while let Some(ch) = chars.next() {
        if c > n {
            return format!("{}{}", &str[..chs], postfix);
        }
        chs += ch.len_utf8();
        c += 1;
    }
    str.to_owned()
}
