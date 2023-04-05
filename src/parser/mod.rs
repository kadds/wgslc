use std::{marker::PhantomData, ops::RangeFrom};

use nom::{sequence::terminated, IResult};


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

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("parse")]
    Parse,
}

// pub enum BasicType {
//     Bool,
//     F32,
//     F16,
//     I32,
//     U32,
// }

pub struct Parser<'a> {
    input: &'a str,
}

impl<'a> Parser<'a> {
    pub fn new(str: &'a str) -> Self {
        Self { input: str }
    }
    pub fn parse(&self) -> Result<ast::ParseResult, Error> {
        let ret = ast::ParseResult::default();
        // try global directive first

        // try global decl
        Ok(ret)
    }
}
