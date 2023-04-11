use test_log::test;
use wgslc;

#[test]
fn test_decl() {
    let input = include_str!("decl.wgsl");
    let parser = wgslc::parser::Parser::new(true);
    let ret = parser.parse(input).unwrap();
    // let json = serde_json::to_string_pretty(&ret).unwrap();
    // println!("{}", json);
}

#[test]
fn test_stmt() {
    let input = include_str!("stmt.wgsl");
    let parser = wgslc::parser::Parser::new(true);
    let ret = parser.parse(input).unwrap();
    // let json = serde_json::to_string_pretty(&ret).unwrap();
    // std::fs::write("test.json", &json);

    // println!("{}", json);
}
