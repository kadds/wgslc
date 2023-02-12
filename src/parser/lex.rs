use std::str::Chars;

use nom::{
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while},
    character::complete::{digit0, digit1},
    combinator::{complete, map, map_opt, map_res},
    error::{make_error, ErrorKind},
    sequence::{preceded, tuple},
    IResult,
};
use super::{token::*, one_of_or_else, incomplete_or_else, number};

fn bool(i: &str) -> IResult<&str, Literal> {
    map_res(alt((tag("true"), tag("false"))), |s| {
        Ok::<_, nom::Err<&str>>(Literal::Bool(s == "true"))
    })(i)
}

fn float_err(i: &str) -> nom::Err<nom::error::Error<&str>> {
    let e: nom::error::Error<&str> = make_error(i, ErrorKind::Float);
    nom::Err::Error(e)
}

// [eE][+-]?[0-9]+
fn float_exponent_part(i: &str) -> IResult<&str, Option<i32>> {
    map_opt(
        preceded(
            tag_no_case::<&str, &str, _>("e"),
            tuple((one_of_or_else("+-", || '+'), digit1)),
        ),
        |(sig, digit)| {
            let val = digit
                .parse::<i32>()
                .map(|v| if sig == '-' { -v } else { v })
                .ok()?;
            Some(Some(val))
        },
    )(i)
}

fn decimal_part(i: &str) -> IResult<&str, (&str, Option<i32>, char)> {
    preceded(
        tag("."),
        map(
             tuple((incomplete_or_else(digit0, || ""), num_exponent_suffix)),
            |(decimal, (exp, suffix))| (decimal, Some(exp.unwrap_or_default()), suffix),
        ),
    )(i)
}

fn num_exponent_suffix(i: &str) -> IResult<&str, (Option<i32>, char)> {
    alt((
        // test suffix
        tuple((float_exponent_part, num_suffix)),
        map_opt(num_suffix, |suffix| Some((None, suffix))),
    ))(i)
}

fn num_suffix(i: &str) -> IResult<&str, char> {
    one_of_or_else("hfiu", || '\0')(i)
}

fn num(i: &str, radix: u32) -> IResult<&str, Literal> {
    if i.is_empty() {
        return Err(nom::Err::Incomplete(nom::Needed::Unknown));
    }
    map_res(
        alt((
            // .023[e+1][f]
            map_opt(decimal_part, |v| Some(("", v))),
            // 4.123[e+1][f]
            // 4.[e-2]
            // 4.
            tuple((digit0, decimal_part)),
            // 4[f]
            // 1[e-3][f]
            tuple((
                digit1,
                map_opt(num_exponent_suffix, |(exp, suffix)| Some(("", exp, suffix))),
            )),
        )),
        |(digit, (decimal, exp, suffix))| {
            number::parse_num_from(digit, decimal, exp, suffix, radix)
                .and_then(|v| {
                    if let Literal::Integer(_) = &v {
                        // integer
                        if digit.len() > 1 && digit.starts_with('0') {
                            return None;
                        }
                    } else {
                        // float
                        if digit.is_empty() && decimal.is_empty() {
                            return None;
                        }
                    }
                    Some(v)
                })
                .ok_or_else(|| float_err(i))
        },
    )(i)
}

fn literal(i: &str) -> IResult<&str, Literal> {
    complete(alt((
        preceded(tag_no_case("0x"), |i| num(i, 16)),
        |i| num(i, 10),
        bool,
    )))(i)
}

fn is_space(c: char) -> bool {
    "\u{0020}\u{0009}\u{000a}\u{000b}\u{000c}\u{000d}\u{0085}\u{200f}\u{2028}\u{2029}"
        .chars()
        .any(|v| v == c)
}

fn space(input: &str) -> IResult<&str, &str> {
    take_while(is_space)(input)
}

fn is_failure<I>(e: nom::Err<nom::error::Error<I>>) -> bool {
    match e {
        nom::Err::Failure(_) => true,
        _ => false,
    }
}

fn is_error<I>(e: nom::Err<nom::error::Error<I>>) -> bool {
    match e {
        nom::Err::Error(_) => true,
        _ => false,
    }
}

fn keyword(i: &str) -> IResult<&str, Keyword> {
    Keyword::try_from(i).map_err(|_| {
        let e: nom::error::Error<&str> = make_error(i, ErrorKind::Alpha);
        nom::Err::Error(e)
    }).map(|v| {
        ("", v)
    })
}

fn reserved_keyword(i: &str) -> IResult<&str, ReservedWord> {
    ReservedWord::try_from(i).map_err(|_| {
        let e: nom::error::Error<&str> = make_error(i, ErrorKind::Alpha);
        nom::Err::Error(e)
    }).map(|v| {
        ("", v)
    })
}

fn check_identifier(mut chars: Chars, start: char) -> bool {
    if !unicode_ident::is_xid_start(start) {
        return false;
    }
    while let Some(ch) = chars.next() {
        if !unicode_ident::is_xid_continue(ch) {
            return false;
        }
    }
    true
}

fn is_identifier(i: &str) -> bool {
    let mut chars = i.chars();
    if let Some(c) = chars.next() {
        if c == '_' {
            if let Some(c) = chars.next() {
                check_identifier(chars, c)
            } else {
                false
            }
        } else {
            check_identifier(chars, c)
        }
    } else {
        false
    }
}


#[cfg(test)]
pub mod tests {
    use super::*;
    macro_rules! assert_ret {
        ($left: expr, $right: expr) => {
            assert_eq!($left.unwrap().1, $right)
        };
    }

    macro_rules! assert_failure {
        ($left: expr) => {
            assert!(is_failure($left.unwrap_err()))
        };
    }

    macro_rules! assert_error {
        ($left: expr) => {
            assert!(is_error($left.unwrap_err()))
        };
    }

    #[test]
    fn ident_test() {
        assert!(is_identifier("_norm"));
        assert!(is_identifier("_a404"));
        assert!(is_identifier("parser_x"));
        assert!(is_identifier("x1"));
        assert!(is_identifier("var"));
        assert!(!is_identifier("__"));
        assert!(!is_identifier("_"));
        assert!(!is_identifier("_0a"));
        assert!(!is_identifier("4"));
        assert!(!is_identifier("4dd"));
    }

    #[test]
    fn keyword_test() {
        assert_ret!(keyword("array"), Keyword::Array);
        assert_ret!(keyword("f32"), Keyword::F32);
        assert_ret!(keyword("mat2x2"), Keyword::Mat2x2);
        assert_ret!(keyword("texture_1d"), Keyword::Texture1d);
        assert_ret!(keyword("if"), Keyword::If);
        assert_ret!(reserved_keyword("auto"), ReservedWord::Auto);

        assert_error!(keyword("array1"));
        assert_error!(keyword("F32"));
    }

    #[test]
    fn literal_test() {
        // bool
        assert_ret!(literal("true"), Literal::Bool(true));
        assert_ret!(literal("false"), Literal::Bool(false));

        // int
        assert_ret!(literal("1u"), Literal::Integer(Integer::U(1)));
        assert_ret!(literal("0"), Literal::Integer(Integer::Abstract(0)));
        assert_ret!(literal("123"), Literal::Integer(Integer::Abstract(123)));
        assert_ret!(literal("0i"), Literal::Integer(Integer::I(0)));
        assert_ret!(
            literal("0x2410"),
            Literal::Integer(Integer::Abstract(0x2410))
        );
        assert_ret!(literal("0X0"), Literal::Integer(Integer::Abstract(0x0)));

        assert_error!(literal("0X"));
        assert_error!(literal("你好"));
        assert_error!(literal("01"));

        // float
        assert_ret!(literal("0.e+4f"), Literal::Float(Float::F32(0f32)));
        assert_ret!(literal("01."), Literal::Float(Float::Abstract(1f64)));
        assert_ret!(literal(".01"), Literal::Float(Float::Abstract(0.01f64)));
        assert_ret!(literal("12.34"), Literal::Float(Float::Abstract(12.34f64)));
        assert_ret!(literal(".0f"), Literal::Float(Float::F32(0f32)));
        assert_ret!(literal(".0h"), Literal::Float(Float::F16(0f32)));
        assert_ret!(literal("1e-3"), Literal::Float(Float::Abstract(0.001f64)));
        assert_ret!(literal("1e+3"), Literal::Float(Float::Abstract(1000f64)));

        // todo: float hex
        assert_ret!(literal("0xa.fp+2"), Literal::Float(Float::Abstract(hexf::hexf64!("0xa.fp+2"))));
        assert_ret!(literal("0x1P+4f"), Literal::Float(Float::F32(hexf::hexf32!("0x1P+4"))));
        // assert_ret!(literal("0X.3"), Literal::Float(Float::Abstract(hexf::hexf64!("0X.3"))));
        assert_ret!(literal("0x3p+2h"), Literal::Float(Float::F16(hexf::hexf32!("0x3p+2"))));
        assert_ret!(literal("0x1.fp-4"), Literal::Float(Float::Abstract(hexf::hexf64!("0x1.fp-4"))));
        assert_ret!(literal("0x3.2p+2h"), Literal::Float(Float::F16(hexf::hexf32!("0x3.2p+2"))));

        assert_error!(literal("ccv"));
    }
}