use std::{
    ops::RangeFrom,
};

use nom::{
    IResult,
};
mod token;
mod lex;
mod number;

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
