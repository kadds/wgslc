use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::str::FromStr;

use super::ast::*;
use super::token::*;
use super::*;
use nom::bytes::complete::take_till;
use nom::character::complete::{char as xchar, one_of};
use nom::combinator::consumed;
use nom::combinator::peek;
use nom::combinator::recognize;
use nom::multi::{many0, many1, many_m_n};
use nom::{
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while},
    character::complete::{digit0, digit1, satisfy},
    combinator::{complete, map, map_opt, map_res, opt},
    error::{make_error, ErrorKind},
    sequence::{delimited, preceded, tuple},
    IResult,
};

pub fn dbg<F, I, O, E>(mut f: F, context: &'static str) -> impl FnMut(I) -> IResult<I, O, E>
where
    F: FnMut(I) -> IResult<I, O, E>,
    I: Debug + Clone,
    E: Debug,
{
    move |i| match f(i.clone()) {
        Err(e) => {
            log::error!("{}: Error({:?}) at: {:?}", context, e, i);
            Err(e)
        }
        a => a,
    }
}

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

static SPACES: phf::Set<char> = phf::phf_set! {
    '\u{0020}',
    '\u{0009}',
    '\u{000a}',
    '\u{000b}',
    '\u{000c}',
    '\u{000d}',
    '\u{0085}',
    '\u{200f}',
    '\u{2028}',
    '\u{2029}'
};

static LINEBREAK: phf::Set<char> = phf::phf_set! {
    '\u{000A}',
    '\u{000B}',
    '\u{000C}',
    '\u{000D}',
    '\u{0085}',
    '\u{2028}',
    '\u{2029}'
};

fn linebreak(i: &str) -> IResult<&str, &str> {
    let mut chars = i.chars();
    if let Some(c) = chars.next() {
        if c == '\u{000A}' {
            // \n\r
            if let Some(c) = chars.next() {
                if c == '\u{000D}' {
                    let (beg, end) = i.split_at(2);
                    return Ok((end, beg));
                }
            }
            let (beg, end) = i.split_at(1);
            return Ok((end, beg));
        } else {
            if LINEBREAK.contains(&c) {
                let (beg, end) = i.split_at(1);
                return Ok((end, beg));
            }
        }
        return Err(nom::Err::Error(make_error(i, ErrorKind::Alpha)));
    }
    Ok(("", i))
}

fn is_space(c: char) -> bool {
    SPACES.contains(&c)
}

fn space(input: &str) -> IResult<&str, &str> {
    take_while(is_space)(input)
}

pub fn space0<T, E: nom::error::ParseError<T>>(input: T) -> IResult<T, T, E>
where
    T: nom::InputTakeAtPosition,
    <T as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
{
    use nom::AsChar;
    input.split_at_position_complete(|item| {
        let c = item.as_char();
        !is_space(c)
    })
}

pub fn lspace0<T, O, E: nom::error::ParseError<T>, G>(g: G) -> impl FnMut(T) -> IResult<T, O, E>
where
    T: nom::InputTakeAtPosition,
    <T as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    G: nom::Parser<T, O, E>,
{
    preceded(space0, g)
}

pub fn lspace1<T, O, E: nom::error::ParseError<T>, G>(g: G) -> impl FnMut(T) -> IResult<T, O, E>
where
    T: nom::InputTakeAtPosition,
    <T as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    G: nom::Parser<T, O, E>,
{
    preceded(space1, g)
}

pub fn rspace0<T, O, E: nom::error::ParseError<T>, G>(g: G) -> impl FnMut(T) -> IResult<T, O, E>
where
    T: nom::InputTakeAtPosition,
    <T as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    G: nom::Parser<T, O, E>,
{
    terminated(g, space0)
}

pub fn rspace1<T, O, E: nom::error::ParseError<T>, G>(g: G) -> impl FnMut(T) -> IResult<T, O, E>
where
    T: nom::InputTakeAtPosition,
    <T as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    G: nom::Parser<T, O, E>,
{
    terminated(g, space1)
}

pub fn space1<T, E: nom::error::ParseError<T>>(input: T) -> IResult<T, T, E>
where
    T: nom::InputTakeAtPosition,
    <T as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
{
    use nom::AsChar;
    input.split_at_position1_complete(
        |item| {
            let c = item.as_char();
            !is_space(c)
        },
        nom::error::ErrorKind::Space,
    )
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
    Keyword::try_from(i)
        .map_err(|_| {
            let e: nom::error::Error<&str> = make_error(i, ErrorKind::Alpha);
            nom::Err::Error(e)
        })
        .map(|v| ("", v))
}

fn reserved_keyword(i: &str) -> IResult<&str, ReservedWord> {
    ReservedWord::try_from(i)
        .map_err(|_| {
            let e: nom::error::Error<&str> = make_error(i, ErrorKind::Alpha);
            nom::Err::Error(e)
        })
        .map(|v| ("", v))
}

fn is_identifier(i: &str) -> bool {
    identifier(i).map(|v| v.0.is_empty()).unwrap_or_default()
}

fn identifier2(i: &str) -> IResult<&str, &str> {
    dbg(
        recognize(tuple((
            satisfy(unicode_ident::is_xid_start),
            take_while(unicode_ident::is_xid_continue),
        ))),
        "identifier2",
    )(i)
}

fn identifier(i: &str) -> IResult<&str, &str> {
    dbg(
        recognize(tuple((opt(xchar('_')), identifier2))),
        "identifier",
    )(i)
}

fn member_identifier(i: &str) -> IResult<&str, &str> {
    dbg(identifier, "member_identifier")(i)
}

fn swizzle_name(i: &str) -> IResult<&str, &str> {
    map_opt(
        alt((
            many_m_n(1, 4, one_of("rgba")),
            many_m_n(1, 4, one_of("xyzw")),
        )),
        |v| Some(&i[..v.len()]),
    )(i)
}

fn component_or_swizzle_specifier(i: &str) -> IResult<&str, ExprId> {
    dbg(
        alt((
            map_opt(
                tuple((
                    preceded(xchar('.'), identifier),
                    opt(component_or_swizzle_specifier),
                )),
                |(ident_name, postfix)| {
                    let spec = PostfixExpression::new_ident(placement_expr_id(), ident_name);
                    let id = spec.into();
                    if let Some(postfix) = postfix {
                        update_expr_for(postfix, |e: &mut PostfixExpression| {
                            e.ident = id;
                        })?;
                    }
                    Some(id)
                },
            ),
            map_opt(
                tuple((
                    delimited(lspace0(xchar('[')), lspace0(expr), lspace0(xchar(']'))),
                    opt(component_or_swizzle_specifier),
                )),
                |(index, postfix)| {
                    let spec = PostfixExpression::new_index(placement_expr_id(), index);
                    let id = spec.into();
                    if let Some(postfix) = postfix {
                        update_expr_for(postfix, |e: &mut PostfixExpression| {
                            e.ident = id;
                        })?;
                    }
                    Some(id)
                },
            ),
        )),
        "component_or_swizzle_specifier",
    )(i)
}

fn primary_expr(i: &str) -> IResult<&str, ExprId> {
    dbg(
        alt((
            map_opt(
                tuple((
                    lspace0(identifier),
                    opt(template_list),
                    opt(delimited(
                        lspace0(xchar('(')),
                        separated_list1_ext_sep(lspace0(xchar(',')), expr),
                        lspace0(xchar(')')),
                    )),
                )),
                |(ident, template, args)| {
                    let ident = template.map_or_else(
                        || IdentExpression::new_ident(ident),
                        |v| IdentExpression::new(ident, v),
                    );
                    Some(if let Some(args) = args {
                        FunctionCallExpression::new(ident, args.into_iter()).into()
                    } else {
                        ident.into()
                    })
                },
            ),
            map_opt(
                delimited(lspace0(xchar('(')), lspace0(expr), lspace0(xchar(')'))),
                |v| Some(ParenExpression::new_paren(v).into()),
            ),
            map_opt(lspace0(consumed(literal)), |(l, r)| {
                Some(LiteralExpression::new(r, l).into())
            }),
        )),
        "primary_expression",
    )(i)
}

fn unary_expr(i: &str) -> IResult<&str, ExprId> {
    dbg(
        lspace0(alt((
            map_opt(
                tuple((one_of("!&*-~"), lspace0(unary_expr))),
                |(op, expr)| {
                    let str = op.to_string();
                    Some(UnaryExpression::new(SynToken::from_str(&str).unwrap(), expr).into())
                },
            ),
            map_opt(
                tuple((primary_expr, opt(component_or_swizzle_specifier))),
                |(e, o)| {
                    Some(if let Some(o) = o {
                        update_expr_for(o, |p: &mut PostfixExpression| {
                            p.ident = e;
                        });
                        o
                    } else {
                        e
                    })
                },
            ),
        ))),
        "unary_expr",
    )(i)
}

fn bitwise_expression_post_unary_expr(i: &str) -> IResult<&str, PartialExprId> {
    dbg(
        map_opt(
            alt((
                many1(tuple((lspace0(tag("&")), lspace0(unary_expr)))),
                many1(tuple((lspace0(tag("^")), lspace0(unary_expr)))),
                many1(tuple((lspace0(tag("|")), lspace0(unary_expr)))),
            )),
            |res| {
                let mut prev = placement_expr_id();
                let mut partial = placement_expr_id();
                for (op, expr) in res {
                    let p = BinaryExpression::new(prev, SynToken::from_str(op).ok()?, expr).into();
                    if partial == placement_expr_id() {
                        partial = p;
                    }
                    prev = p;
                }
                Some(PartialExprId { top: prev, partial })
            },
        ),
        "bitwise_expression_post_unary_expression",
    )(i)
}

fn multiplicative_operator(i: &str) -> IResult<&str, &str> {
    rspace0(lspace0(alt((tag("%"), tag("*"), tag("/")))))(i)
}

fn additive_operator(i: &str) -> IResult<&str, &str> {
    rspace0(lspace0(alt((tag("+"), tag("-")))))(i)
}

fn shift_expression_post_unary_expr(i: &str) -> IResult<&str, PartialExprId> {
    dbg(
        alt((
            map_opt(
                tuple((
                    many0(tuple((multiplicative_operator, lspace0(unary_expr)))),
                    many0(tuple((
                        additive_operator,
                        lspace0(unary_expr),
                        many0(tuple((multiplicative_operator, lspace0(unary_expr)))),
                    ))),
                )),
                |(a, b)| {
                    let mut expr = placement_expr_id();
                    let mut beg = None;
                    for (op, right) in a {
                        expr =
                            BinaryExpression::new(expr, SynToken::from_str(op).ok()?, right).into();
                        if beg.is_none() {
                            beg = Some(expr);
                        }
                    }
                    for (op, right, ext) in b {
                        let mut expr2 = right;
                        for (op, right) in ext {
                            expr2 =
                                BinaryExpression::new(expr2, SynToken::from_str(op).ok()?, right)
                                    .into();
                        }

                        expr =
                            BinaryExpression::new(expr, SynToken::from_str(op).ok()?, expr2).into();

                        if beg.is_none() {
                            beg = Some(expr);
                        }
                    }
                    if expr == placement_expr_id() {
                        return Some(PartialExprId::new_empty());
                    }
                    Some(PartialExprId {
                        top: expr,
                        partial: beg.unwrap(),
                    })
                },
            ),
            map_opt(
                alt((
                    tuple((lspace0(tag("<<")), unary_expr)),
                    tuple((lspace0(tag(">>")), unary_expr)),
                )),
                |(_tag, _expr)| Some(PartialExprId::new_empty()),
            ),
        )),
        "shift_expression_post_unary_expression",
    )(i)
}

fn relational_expression_post_unary_expr(i: &str) -> IResult<&str, PartialExprId> {
    dbg(
        map_opt(
            tuple((
                shift_expression_post_unary_expr,
                opt(tuple((
                    lspace0(alt((
                        tag("!="),
                        tag("=="),
                        tag(">="),
                        tag("<="),
                        tag(">"),
                        tag("<"),
                    ))),
                    unary_expr,
                    shift_expression_post_unary_expr,
                ))),
            )),
            |(e, opt)| {
                log::info!("opt {:?}, {:?}", e, opt);
                if let Some((op, expr, _e2)) = opt {
                    let top =
                        BinaryExpression::new(e.top, SynToken::from_str(op).ok()?, expr).into();
                    if e.is_empty() {
                        Some(PartialExprId { top, partial: top })
                    } else {
                        Some(PartialExprId {
                            top,
                            partial: e.partial,
                        })
                    }
                } else {
                    Some(e)
                }
            },
        ),
        "relational_expression_post_unary_expression",
    )(i)
}

fn expr(i: &str) -> IResult<&str, ExprId> {
    dbg(
        map_opt(
            tuple((
                unary_expr,
                alt((
                    map_opt(
                        tuple((
                            relational_expression_post_unary_expr,
                            opt(lspace0(alt((
                                preceded(
                                    peek(tag("&&")),
                                    many1(tuple((
                                        lspace0(tag("&&")),
                                        lspace0(unary_expr),
                                        lspace0(relational_expression_post_unary_expr),
                                    ))),
                                ),
                                preceded(
                                    peek(tag("||")),
                                    many1(tuple((
                                        lspace0(tag("||")),
                                        lspace0(unary_expr),
                                        lspace0(relational_expression_post_unary_expr),
                                    ))),
                                ),
                            )))),
                        )),
                        |(r, extends)| {
                            if r.is_empty() && extends.is_none() {
                                return None;
                            }
                            log::error!("dsfsdfsdf  {:#?}  \n {:#?}", r, extends);
                            if let Some(extends) = extends {
                                let mut partial = r.partial;
                                let mut prev_expr = r.top;
                                for (tag, l, relate) in extends {
                                    let op = SynToken::from_str(tag).ok()?;
                                    let expr = if relate.is_empty() {
                                        BinaryExpression::new(prev_expr, op, l)
                                    } else {
                                        update_expr_for(
                                            relate.partial,
                                            |expr: &mut BinaryExpression| {
                                                expr.left = l;
                                            },
                                        );
                                        BinaryExpression::new(prev_expr, op, relate.top)
                                    }
                                    .into();
                                    if prev_expr == placement_expr_id() {
                                        partial = expr;
                                    }
                                    prev_expr = expr;
                                }
                                return Some(PartialExprId {
                                    top: prev_expr,
                                    partial,
                                });
                            }
                            Some(r)
                        },
                    ),
                    bitwise_expression_post_unary_expr,
                    map_opt(tag(""), |_| Some(PartialExprId::new_empty())),
                )),
            )),
            |(u, e)| {
                if !e.is_empty() {
                    log::info!("extend expr {:#?}", e);
                    let ret = update_expr_for(e.partial, |expr: &mut BinaryExpression| {
                        expr.left = u;
                    });
                    if let Some(()) = ret {
                        return Some(e.top);
                    }
                }
                Some(u)
            },
        ),
        "expr",
    )(i)
}

fn template_list(i: &str) -> IResult<&str, ExprId> {
    map_opt(
        delimited(
            lspace0(xchar('<')),
            separated_list1_ext_sep(terminated(lspace0(xchar(',')), space0), expr),
            lspace0(xchar('>')),
        ),
        |exprs| {
            let mut prev_expr = placement_expr_id();
            let mut start_expr = placement_expr_id();
            for expr in exprs {
                let c = ConcatExpression::new_end(expr).into();
                if start_expr == placement_expr_id() {
                    start_expr = c;
                }
                if prev_expr != placement_expr_id() {
                    update_expr_for(prev_expr, |e: &mut ConcatExpression| {
                        e.right = Some(c);
                    });
                }
                prev_expr = c;
            }
            Some(ListExpression::new_comma(start_expr).into())
        },
    )(i)
}

fn type_specifier(i: &str) -> IResult<&str, Ty<'_>> {
    map_opt(tuple((identifier, opt(template_list))), |(ident, list)| {
        if let Some(template) = list {
            Some(Ty::IdentTemplate((ident, template)))
        } else {
            Some(Ty::Ident(ident))
        }
    })(i)
}

fn struct_member(i: &str) -> IResult<&str, StructMember> {
    map_opt(
        tuple((
            many0(rspace1(attribute)),
            lspace0(member_identifier),
            lspace0(xchar(':')),
            lspace0(type_specifier),
        )),
        |(attrs, member_name, _, ty)| {
            Some(StructMember {
                ident: member_name,
                ty,
                attrs,
            })
        },
    )(i)
}

fn struct_body_decl(i: &str) -> IResult<&str, StructDecl> {
    map_opt(
        delimited(
            lspace0(xchar('{')),
            separated_list1_ext_sep(lspace0(xchar(',')), struct_member),
            lspace0(xchar('}')),
        ),
        |members| Some(StructDecl { name: "", members }),
    )(i)
}

fn optionally_typed_ident(i: &str) -> IResult<&str, OptionallyTypedIdent<'_>> {
    map_opt(
        tuple((
            identifier,
            opt(preceded(lspace0(xchar(':')), lspace0(type_specifier))),
        )),
        |(ident, ty)| Some(OptionallyTypedIdent { name: ident, ty }),
    )(i)
}

fn param(i: &str) -> IResult<&str, Param<'_>> {
    map_opt(
        tuple((
            identifier,
            preceded(lspace0(xchar(':')), lspace0(type_specifier)),
        )),
        |(ident, ty)| Some(Param { name: ident, ty }),
    )(i)
}

fn function_inputs(i: &str) -> IResult<&str, Vec<Param<'_>>> {
    separated_list1_ext_sep(lspace0(xchar(',')), lspace0(param))(i)
}

fn function_header(i: &str) -> IResult<&str, FunctionDecl> {
    map_opt(
        tuple((
            lspace0(tag("fn")),
            lspace1(identifier),
            delimited(
                lspace0(xchar('(')),
                lspace0(function_inputs),
                lspace0(xchar(')')),
            ),
            opt(preceded(
                lspace0(tag("->")),
                tuple((many0(rspace1(attribute)), lspace0(type_specifier))),
            )),
        )),
        |(_, name, inputs, output)| {
            Some(FunctionDecl {
                name,
                inputs,
                output,
                ast: None,
                attrs: vec![],
            })
        },
    )(i)
}

fn diagnostic_control(i: &str) -> IResult<&str, DiagnosticControl> {
    map_opt(
        delimited(
            lspace0(xchar('(')),
            tuple((
                lspace0(alt((tag("error"), tag("info"), tag("off"), tag("warning")))),
                identifier,
                opt(lspace0(xchar(','))),
            )),
            lspace0(xchar(')')),
        ),
        |(n, i, _)| Some(DiagnosticControl { name: n, ident: i }),
    )(i)
}

fn attribute(i: &str) -> IResult<&str, Attribute<'_>> {
    let (rest, result) = preceded(xchar('@'), lspace0(identifier))(i)?;
    let attr = AttributeType::from_str(result).map_err(|_| {
        let e: nom::error::Error<&str> = make_error(i, ErrorKind::Alpha);
        nom::Err::Error(e)
    })?;
    if let Some((min, max)) = attr.extendable() {
        if attr == AttributeType::Diagnostic {
            let (rest, result) = diagnostic_control(rest)?;
            return Ok((
                rest,
                Attribute {
                    ty: attr,
                    diagnostic_control: Some(result),
                    exprs: vec![],
                },
            ));
        }
        map_opt(
            delimited(
                lspace0(xchar('(')),
                tuple((
                    lspace0(expr),
                    many_m_n(
                        min - 1,
                        max - 1,
                        preceded(lspace0(xchar(',')), lspace0(expr)),
                    ),
                )),
                tuple((opt(lspace0(xchar(','))), lspace0(xchar(')')))),
            ),
            |(expr0, mut exprs)| {
                exprs.insert(0, expr0);
                Some(Attribute {
                    ty: attr,
                    exprs,
                    diagnostic_control: None,
                })
            },
        )(rest)
    } else {
        Ok((
            rest,
            Attribute {
                ty: attr,
                exprs: vec![],
                diagnostic_control: None,
            },
        ))
    }
}

fn attributes(i: &str) -> IResult<&str, Vec<Attribute<'_>>> {
    many0(rspace1(attribute))(i)
}

fn statement(_i: &str) -> IResult<&str, Statement<'_>> {
    todo!()
}

fn line_comment(i: &str) -> IResult<&str, &str> {
    map_opt(
        tuple((tag("//"), take_till(|c| LINEBREAK.contains(&c)), linebreak)),
        |(_, v, _)| Some(v),
    )(i)
}

fn block_comment_body(i: &str) -> IResult<&str, &str> {
    let mut chars = i.chars();
    let mut deep = 0;
    let mut last_ch = '\0';
    let mut lens = 0;
    while let Some(c) = chars.next() {
        if c == '*' && last_ch == '/' {
            deep += 1;
        } else if c == '/' && last_ch == '*' {
            if deep == 0 {
                lens -= 1;
                break;
            }
            deep -= 1;
            last_ch = '\0';
        } else {
            last_ch = c;
        }
        lens += c.len_utf8();
    }

    let (beg, end) = i.split_at(lens);
    Ok((end, beg))
}

fn block_comment(i: &str) -> IResult<&str, &str> {
    map_opt(
        tuple((tag("/*"), block_comment_body, tag("*/"))),
        |(_, v, _)| Some(v),
    )(i)
}

fn comment(i: &str) -> IResult<&str, &str> {
    alt((line_comment, block_comment))(i)
}

fn compound_statement(i: &str) -> IResult<&str, CompoundStatement> {
    map_opt(
        tuple((
            lspace0(attributes),
            delimited(lspace0(xchar('{')), many0(statement), lspace0(xchar('}'))),
        )),
        |(attrs, statements)| Some(CompoundStatement { attrs, statements }),
    )(i)
}

fn assignment_statement(i: &str) -> IResult<&str, AssignmentStatement> {
    todo!()
}

fn variable_decl(i: &str) -> IResult<&str, GlobalVariableDecl> {
    map_opt(
        tuple((
            tag("var"),
            opt(lspace0(template_list)),
            lspace1(optionally_typed_ident),
        )),
        |(_, list, ident)| {
            Some(GlobalVariableDecl {
                template_list: list,
                ident,
                equals: None,
                ..Default::default()
            })
        },
    )(i)
}

fn global_override_value_decl(i: &str) -> IResult<&str, GlobalOverrideValueDecl> {
    map_opt(
        tuple((
            lspace0(attributes),
            lspace0(tag("override")),
            lspace1(optionally_typed_ident),
            opt(preceded(lspace0(xchar('=')), lspace0(expr))),
        )),
        |(attrs, _, ident, expr)| {
            Some(GlobalOverrideValueDecl {
                ident,
                equals: expr,
                attrs,
            })
        },
    )(i)
}

fn global_const_value_decl(i: &str) -> IResult<&str, GlobalConstValueDecl> {
    map_opt(
        preceded(
            tag("const"),
            tuple((
                lspace1(optionally_typed_ident),
                preceded(lspace0(xchar('=')), lspace0(expr)),
            )),
        ),
        |(ident, expr)| {
            Some(GlobalConstValueDecl {
                ident,
                equals: expr,
            })
        },
    )(i)
}

fn global_variable_decl(i: &str) -> IResult<&str, GlobalVariableDecl> {
    map_opt(
        tuple((
            lspace0(attributes),
            lspace0(variable_decl),
            opt(preceded(lspace0(xchar('=')), lspace0(expr))),
        )),
        |(attrs, mut va, exp)| {
            va.equals = exp;
            va.attrs = attrs;
            Some(va)
        },
    )(i)
}

fn global_value_decl(i: &str) -> IResult<&str, GlobalValueDecl> {
    alt((
        map_opt(global_override_value_decl, |v| {
            Some(GlobalValueDecl::GlobalOverrideValueDecl(v))
        }),
        map_opt(global_const_value_decl, |v| {
            Some(GlobalValueDecl::GlobalConstValueDecl(v))
        }),
    ))(i)
}

fn type_alias_decl(i: &str) -> IResult<&str, TypeAliasDecl> {
    map_opt(
        tuple((
            lspace0(tag("alias")),
            lspace1(identifier),
            lspace0(tag("=")),
            lspace0(type_specifier),
        )),
        |(_, name, _, ty)| Some(TypeAliasDecl { name, ty }),
    )(i)
}

fn struct_decl(i: &str) -> IResult<&str, StructDecl> {
    map_opt(
        tuple((
            preceded(tag("struct"), lspace1(identifier)),
            struct_body_decl,
        )),
        |(name, mut struct_members)| {
            struct_members.name = name;
            Some(struct_members)
        },
    )(i)
}

fn function_decl(i: &str) -> IResult<&str, FunctionDecl> {
    map_opt(
        tuple((attributes, function_header, compound_statement)),
        |(attrs, mut decl, _detail)| {
            decl.attrs = attrs;
            Some(decl)
        },
    )(i)
}

fn const_assert_statement(i: &str) -> IResult<&str, ConstAssertStatement> {
    map_opt(preceded(tag("const_assert"), lspace1(expr)), |expr| {
        Some(ConstAssertStatement {
            expr,
            _pd: PhantomData::default(),
        })
    })(i)
}

#[allow(unused)]
fn enable_directive(i: &str) -> IResult<&str, &str> {
    delimited(
        tag("enable"),
        delimited(space1, identifier, space0),
        tag(";"),
    )(i)
}

#[allow(unused)]
fn global_decl(i: &str) -> IResult<&str, GlobalDecl<'_>> {
    preceded(
        space0,
        alt((
            map_opt(xchar(';'), |_| Some(GlobalDecl::None)),
            map_opt(terminated(global_variable_decl, lspace0(xchar(';'))), |v| {
                Some(GlobalDecl::GlobalVariableDecl(v))
            }),
            map_opt(terminated(global_value_decl, lspace0(xchar(';'))), |v| {
                Some(GlobalDecl::GlobalValueDecl(v))
            }),
            map_opt(terminated(type_alias_decl, lspace0(xchar(';'))), |v| {
                Some(GlobalDecl::TypeAliasDecl(v))
            }),
            map_opt(struct_decl, |v| Some(GlobalDecl::StructDecl(v))),
            map_opt(function_decl, |v| Some(GlobalDecl::FunctionDecl(v))),
            map_opt(
                terminated(const_assert_statement, lspace0(xchar(';'))),
                |v| Some(GlobalDecl::ConstAssertStatement(v)),
            ),
        )),
    )(i)
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use pretty_assertions::assert_eq;
    use test_log::test;
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
    fn expr_test() {
        assert_ret!(
            shift_expression_post_unary_expr("-b*c+d*e").map(|v| (v.0, v.1.top)),
            BinaryExpression::new(
                BinaryExpression::new(
                    placement_expr_id(),
                    SynToken::Minus,
                    BinaryExpression::new(
                        IdentExpression::new_ident("b"),
                        SynToken::Star,
                        IdentExpression::new_ident("c"),
                    ),
                ),
                SynToken::Plus,
                BinaryExpression::new(
                    IdentExpression::new_ident("d"),
                    SynToken::Star,
                    IdentExpression::new_ident("e"),
                ),
            )
            .into()
        );

        assert_ret!(
            expr("1"),
            LiteralExpression::new(Integer::Abstract(1).into(), "1").into()
        );

        assert_ret!(
            expr("a-b*c+d"),
            BinaryExpression::new(
                BinaryExpression::new(
                    IdentExpression::new_ident("a"),
                    SynToken::Minus,
                    BinaryExpression::new(
                        IdentExpression::new_ident("b"),
                        SynToken::Star,
                        IdentExpression::new_ident("c")
                    )
                ),
                SynToken::Plus,
                IdentExpression::new_ident("d")
            )
            .into()
        );

        assert_ret!(
            expr("1-a+f(1)"),
            BinaryExpression::new(
                BinaryExpression::new(
                    LiteralExpression::new(Integer::Abstract(1).into(), "1"),
                    SynToken::Minus,
                    IdentExpression::new_ident("a"),
                ),
                SynToken::Plus,
                FunctionCallExpression::new_slice(
                    IdentExpression::new_ident("f").into(),
                    &[LiteralExpression::new(Integer::Abstract(1).into(), "1").into()]
                )
            )
            .into()
        );
    }

    #[test]
    fn expr_unary_test() {
        assert_ret!(
            expr("!a"),
            UnaryExpression::new(SynToken::Bang, IdentExpression::new_ident("a")).into()
        );

        assert_ret!(
            expr("-a"),
            UnaryExpression::new(SynToken::Minus, IdentExpression::new_ident("a")).into()
        );
    }

    #[test]
    fn expr_logical_compare_test() {
        assert_ret!(
            expr("a<=b"),
            BinaryExpression::new(
                IdentExpression::new_ident("a"),
                SynToken::LessThanEqual,
                IdentExpression::new_ident("b")
            )
            .into()
        );

        assert_ret!(
            expr("a==b&&c>d"),
            BinaryExpression::new(
                BinaryExpression::new(
                    IdentExpression::new_ident("a"),
                    SynToken::EqualEqual,
                    IdentExpression::new_ident("b")
                ),
                SynToken::AndAnd,
                BinaryExpression::new(
                    IdentExpression::new_ident("c"),
                    SynToken::GreaterThan,
                    IdentExpression::new_ident("d")
                ),
            )
            .into()
        );
    }

    #[test]
    fn expr_logical_test() {
        assert_ret!(
            expr("a&&d&&e"),
            BinaryExpression::new(
                BinaryExpression::new(
                    IdentExpression::new_ident("a"),
                    SynToken::AndAnd,
                    IdentExpression::new_ident("d")
                ),
                SynToken::AndAnd,
                IdentExpression::new_ident("e")
            )
            .into()
        );

        assert_ret!(
            expr("a||(d&&e)"),
            BinaryExpression::new(
                IdentExpression::new_ident("a"),
                SynToken::OrOr,
                ParenExpression::new_paren(BinaryExpression::new(
                    IdentExpression::new_ident("d"),
                    SynToken::AndAnd,
                    IdentExpression::new_ident("e")
                ))
            )
            .into()
        );

        assert_ret!(
            expr("a||((b||c)&&d)"),
            BinaryExpression::new(
                IdentExpression::new_ident("a"),
                SynToken::OrOr,
                ParenExpression::new_paren(BinaryExpression::new(
                    ParenExpression::new_paren(BinaryExpression::new(
                        IdentExpression::new_ident("b"),
                        SynToken::OrOr,
                        IdentExpression::new_ident("c"),
                    ),),
                    SynToken::AndAnd,
                    IdentExpression::new_ident("d")
                ))
            )
            .into()
        );

        assert_ret!(
            expr("x&(y^(z|w))"),
            BinaryExpression::new(
                IdentExpression::new_ident("x"),
                SynToken::And,
                ParenExpression::new_paren(BinaryExpression::new(
                    IdentExpression::new_ident("y"),
                    SynToken::Xor,
                    ParenExpression::new_paren(BinaryExpression::new(
                        IdentExpression::new_ident("z"),
                        SynToken::Or,
                        IdentExpression::new_ident("w")
                    ))
                ))
            )
            .into()
        );
    }

    #[test]
    fn const_assert_statement_test() {
        let (_, c) = const_assert_statement("const_assert a!=y").unwrap();
        assert_eq!(
            c.expr,
            BinaryExpression::new(
                IdentExpression::new_ident("a"),
                SynToken::NotEqual,
                IdentExpression::new_ident("y")
            )
            .into()
        );
    }

    #[test]
    fn function_header_test() {
        assert_ret!(
            function_header("fn add_two(i: i32, b: f32) -> i32"),
            FunctionDecl {
                name: "add_two",
                inputs: vec![
                    Param {
                        name: "i",
                        ty: Ty::Ident("i32")
                    },
                    Param {
                        name: "b",
                        ty: Ty::Ident("f32")
                    }
                ],
                output: Some((vec!(), Ty::Ident("i32"))),
                ast: None,
                attrs: vec!(),
            }
        );
    }

    #[test]
    fn global_variable_decl_test() {
        assert_ret!(
            global_variable_decl("var i: i32"),
            GlobalVariableDecl {
                template_list: None,
                ident: OptionallyTypedIdent {
                    name: "i",
                    ty: Some(Ty::Ident("i32"))
                },
                equals: None,
                attrs: vec!(),
            }
        );
    }

    #[test]
    fn global_value_decl_test() {
        assert_ret!(
            global_const_value_decl("const b : i32 = 4"),
            GlobalConstValueDecl {
                ident: OptionallyTypedIdent {
                    name: "b",
                    ty: Some(Ty::Ident("i32"))
                },
                equals: LiteralExpression::new(Integer::Abstract(4).into(), "4").into(),
            }
        );

        assert_ret!(
            global_override_value_decl("override width: f32"),
            GlobalOverrideValueDecl {
                ident: OptionallyTypedIdent {
                    name: "width",
                    ty: Some(Ty::Ident("f32"))
                },
                equals: None,
                attrs: vec!(),
            }
        );
    }

    #[test]
    fn type_alias_decl_test() {
        let (_, decl) = type_alias_decl("alias Arr = array<i32>").unwrap();
        assert_eq!(
            decl,
            TypeAliasDecl {
                name: "Arr",
                ty: Ty::IdentTemplate((
                    "array",
                    ListExpression::new_comma(ConcatExpression::new_end(
                        IdentExpression::new_ident("i32"),
                    ))
                    .into()
                ))
                .into(),
            }
        );

        assert_ret!(
            type_alias_decl("alias single = f32"),
            TypeAliasDecl {
                name: "single",
                ty: Ty::Ident("f32")
            }
        );
    }

    #[test]
    fn comment_test() {
        assert_ret!(comment("//test "), "test ");
        assert_ret!(comment("/*test*/"), "test");
        assert_ret!(comment("/*test/*a*/*/"), "test/*a*/");
        assert!(comment(
            r#"/*
     This is a block comment
                that spans lines.
                /* Block comments can nest.
                 */
                But all block comments must terminate.
               */
        */"#
        )
        .is_ok())
    }

    #[test]
    fn attribute_test() {
        assert_ret!(
            attribute("@const"),
            Attribute {
                ty: AttributeType::Const,
                diagnostic_control: None,
                exprs: vec![],
            }
        );

        assert_ret!(
            attribute("@size(16)"),
            Attribute {
                ty: AttributeType::Size,
                diagnostic_control: None,
                exprs: vec![LiteralExpression::new(Integer::Abstract(16).into(), "16").into()],
            }
        );
    }

    #[test]
    fn struct_decl_test() {
        assert_eq!(
            separated_list1_ext_sep(tag(","), identifier)("a,b,"),
            Ok(("", vec!["a", "b"]))
        );

        let struct_str = r#"struct Data {
          a: i32,
          b: vec2<f32>,
          c: array<i32,10>,
          d: array<vec4<f32>>,
        }"#;
        assert_ret!(
            struct_decl(struct_str),
            StructDecl {
                name: "Data",
                members: vec![
                    StructMember {
                        ident: "a",
                        ty: Ty::Ident("i32"),
                        attrs: vec!(),
                    },
                    StructMember {
                        ident: "b",
                        ty: Ty::IdentTemplate((
                            "vec2",
                            ListExpression::new_comma(ConcatExpression::new_end(
                                IdentExpression::new_ident("f32")
                            ))
                            .into()
                        )),
                        attrs: vec!(),
                    },
                    StructMember {
                        ident: "c",
                        ty: Ty::IdentTemplate((
                            "array",
                            ListExpression::new_comma(ConcatExpression::new(
                                IdentExpression::new_ident("i32"),
                                ConcatExpression::new_end(LiteralExpression::new(
                                    Integer::Abstract(10).into(),
                                    "10"
                                ))
                            ))
                            .into()
                        )),
                        attrs: vec!(),
                    },
                    StructMember {
                        ident: "d",
                        ty: Ty::IdentTemplate((
                            "array",
                            ListExpression::new_comma(ConcatExpression::new_end(
                                IdentExpression::new(
                                    "vec4",
                                    ListExpression::new_comma(ConcatExpression::new_end(
                                        IdentExpression::new_ident("f32"),
                                    ))
                                ),
                            ))
                            .into()
                        )),
                        attrs: vec!(),
                    },
                ]
            }
        )
    }

    #[test]
    fn directive_test() {
        assert_ret!(enable_directive("enable f16;"), "f16");
        assert_ret!(enable_directive("enable   aka ;"), "aka");
        assert_error!(enable_directive("enableaka;"));
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
        assert!(is_identifier("Δέλτα"));
        assert!(is_identifier("réflexion"));
        assert!(is_identifier("गुलाबी"));

        assert_ret!(identifier("abc_1"), "abc_1");
        assert_ret!(identifier("abc_1;"), "abc_1");
        assert_ret!(identifier("_abc_1;"), "_abc_1");
        assert_error!(identifier("_;"));
        assert_error!(identifier("__;"));
        assert_error!(identifier("1;"));
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
        // assert_ret!(
        //     literal("0xa.fp+2"),
        //     Literal::Float(Float::Abstract(hexf::hexf64!("0xa.fp+2")))
        // );
        // assert_ret!(
        //     literal("0x1P+4f"),
        //     Literal::Float(Float::F32(hexf::hexf32!("0x1P+4")))
        // );
        // // assert_ret!(literal("0X.3"), Literal::Float(Float::Abstract(hexf::hexf64!("0X.3"))));
        // assert_ret!(
        //     literal("0x3p+2h"),
        //     Literal::Float(Float::F16(hexf::hexf32!("0x3p+2")))
        // );
        // assert_ret!(
        //     literal("0x1.fp-4"),
        //     Literal::Float(Float::Abstract(hexf::hexf64!("0x1.fp-4")))
        // );
        // assert_ret!(
        //     literal("0x3.2p+2h"),
        //     Literal::Float(Float::F16(hexf::hexf32!("0x3.2p+2")))
        // );

        // assert_error!(literal("ccv"));
    }
}
