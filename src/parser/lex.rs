use std::str::{Chars, FromStr};

use super::token::*;
use super::*;
use nom::bytes::complete::take_till;
use nom::character::complete::char as xchar;
use nom::multi::many0;
use nom::{
    branch::alt,
    bytes::complete::{tag, tag_no_case, take_while},
    character::complete::{digit0, digit1, satisfy},
    combinator::{complete, map, map_opt, map_res, opt},
    error::{make_error, ErrorKind},
    sequence::{delimited, preceded, tuple},
    IResult,
};

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
    map_opt::<&str, _, _, _, _, _>(
        tuple((
            satisfy(unicode_ident::is_xid_start),
            take_while(unicode_ident::is_xid_continue),
        )),
        |(start, continues)| Some(&i[..(1 + continues.len())]),
    )(i)
}

fn identifier(i: &str) -> IResult<&str, &str> {
    map_opt(tuple((opt(xchar('_')), identifier2)), |(prefix, ident)| {
        Some(if prefix.is_some() {
            &i[..(1 + ident.len())]
        } else {
            ident
        })
    })(i)
}

fn member_identifier(i: &str) -> IResult<&str, &str> {
    identifier(i)
}

fn enable_directive(i: &str) -> IResult<&str, &str> {
    delimited(
        tag("enable"),
        delimited(space1, identifier, space0),
        tag(";"),
    )(i)
}

fn expr(i: &str) -> IResult<&str, Expression<'_>> {
    todo!()
}

fn template_list(i: &str) -> IResult<&str, TemplateList<'_>> {
    map_opt(
        delimited(
            lspace0(xchar('<')),
            separated_list1_ext_sep(
                terminated(lspace0(xchar(',')), space0),
                take_till(|c| c == '>' || c == ','),
            ),
            lspace0(xchar('>')),
        ),
        |vars| Some(TemplateList { vars }),
    )(i)
}

fn type_specifier(i: &str) -> IResult<&str, Ty<'_>> {
    map_opt(tuple((identifier, opt(template_list))), |(ident, list)| {
        if let Some(list) = list {
            Some(Ty::TemplateIdent((ident, list)))
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

fn attribute(i: &str) -> IResult<&str, Attribute<'_>> {
    let (rest, result) = preceded(xchar('@'), lspace0(identifier))(i)?;
    let attr = AttributeType::from_str(result).map_err(|_| {
        let e: nom::error::Error<&str> = make_error(i, ErrorKind::Alpha);
        nom::Err::Error(e)
    })?;
    if attr.is_extendable() {
        // if attr == AttributeType::Diagnostic {
        //     Ok((rest, Attribute {ty: attr, diagnostic_control}))
        map_opt(
            delimited(
                lspace0(xchar('(')),
                terminated(expr, opt(lspace0(xchar(',')))),
                lspace0(xchar(')')),
            ),
            |expr| {
                Some(Attribute {
                    ty: attr,
                    expr: Some(expr),
                    diagnostic_control: None,
                })
            },
        )(rest)
    } else {
        Ok((
            rest,
            Attribute {
                ty: attr,
                expr: None,
                diagnostic_control: None,
            },
        ))
    }
}

fn attributes(i: &str) -> IResult<&str, Vec<Attribute<'_>>> {
    many0(attribute)(i)
}

fn statement(i: &str) -> IResult<&str, Statement<'_>> {
    todo!()
}

fn compound_statement(i: &str) -> IResult<&str, CompoundStatement<'_>> {
    map_opt(
        tuple((
            many0(rspace1(attribute)),
            delimited(lspace0(xchar('{')), many0(statement), lspace0(xchar('}'))),
        )),
        |(attrs, statements)| Some(CompoundStatement { attrs, statements }),
    )(i)
}

fn variable_decl(i: &str) -> IResult<&str, GlobalVariableDecl> {
    map_opt(
        tuple((
            tag("var"),
            opt(lspace1(template_list)),
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
            many0(rspace1(attribute)),
            lspace1(tag("override")),
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
                opt(preceded(lspace0(xchar('=')), lspace0(expr))),
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
            many0(rspace1(attribute)),
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
        tuple((
            many0(rspace1(attribute)),
            function_header,
            compound_statement,
        )),
        |(attrs, mut decl, detail)| {
            decl.attrs = attrs;
            Some(decl)
        },
    )(i)
}

fn const_assert_statement(i: &str) -> IResult<&str, ConstAssertStatement> {
    map_opt(preceded(tag("const_assert"), lspace1(expr)), |v| {
        Some(ConstAssertStatement { expr: v })
    })(i)
}

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
    fn expr_test() {}

    #[test]
    fn const_assert_statement_test() {
        // assert_ret!(const_assert_statement("const_assert a!=y"), ConstAssertStatement {expr: Expression {}})
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
    fn type_alias_decl_test() {
        assert_ret!(
            type_alias_decl("alias Arr = array<i32, 5>"),
            TypeAliasDecl {
                name: "Arr",
                ty: Ty::TemplateIdent((
                    "array",
                    TemplateList {
                        vars: vec!["i32", "5"]
                    }
                ))
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
    fn struct_decl_test() {
        assert_eq!(
            separated_list1_ext_sep(tag(","), identifier)("a,b,"),
            Ok(("", vec!["a", "b"]))
        );

        let struct_str = r#"struct Data {
  a: i32,
  b: vec2<f32>,
  c: array<i32,10>,
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
                        ty: Ty::TemplateIdent(("vec2", TemplateList { vars: vec!["f32"] })),
                        attrs: vec!(),
                    },
                    StructMember {
                        ident: "c",
                        ty: Ty::TemplateIdent((
                            "array",
                            TemplateList {
                                vars: vec!["i32", "10"]
                            }
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
        assert_ret!(
            literal("0xa.fp+2"),
            Literal::Float(Float::Abstract(hexf::hexf64!("0xa.fp+2")))
        );
        assert_ret!(
            literal("0x1P+4f"),
            Literal::Float(Float::F32(hexf::hexf32!("0x1P+4")))
        );
        // assert_ret!(literal("0X.3"), Literal::Float(Float::Abstract(hexf::hexf64!("0X.3"))));
        assert_ret!(
            literal("0x3p+2h"),
            Literal::Float(Float::F16(hexf::hexf32!("0x3p+2")))
        );
        assert_ret!(
            literal("0x1.fp-4"),
            Literal::Float(Float::Abstract(hexf::hexf64!("0x1.fp-4")))
        );
        assert_ret!(
            literal("0x3.2p+2h"),
            Literal::Float(Float::F16(hexf::hexf32!("0x3.2p+2")))
        );

        assert_error!(literal("ccv"));
    }
}
