use wgslc;
use test_log::test;

#[test]
fn test_decl() {
    let input = include_str!("decl.wgsl");
    let parser =wgslc::parser::Parser::new(true);
    parser.parse(input).unwrap();
}