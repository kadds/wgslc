use super::token::{Float, Integer, Literal};

pub fn parse_num_from(
    digit: &str,
    decimal: &str,
    exp: Option<i32>,
    suffix: char,
    radix: u32,
) -> Option<Literal> {
    let digit = if digit.is_empty() { "0" } else { digit };
    let (decimal, bits) = if decimal.is_empty() {
        ("0", 0)
    } else {
        (decimal, decimal.len())
    };
    if suffix == 'f' || suffix == 'h' || exp.is_some() {
        // float
        let digit: u32 = digit.parse::<u32>().ok()?;
        let decimal = decimal.parse::<u32>().ok()?;

        let val = digit * 10u32.pow(bits as u32) + decimal;
        let exp = exp.unwrap() - bits as i32;

        let fval = if exp > 0 {
            val as f64 * 10_u32.pow(exp as u32) as f64
        } else if exp < 0 {
            val as f64 / 10_u32.pow(-exp as u32) as f64
        } else {
            val as f64
        };
        let f = if suffix == 'f' {
            Float::F32(fval as f32)
        } else if suffix == 'h' {
            Float::F16(fval as f32)
        } else {
            Float::Abstract(fval)
        };
        Some(Literal::Float(f))
    } else {
        // integer
        Some(Literal::Integer(
            if suffix == 'i' {
                i32::from_str_radix(digit, radix).map(Integer::I)
            } else if suffix == 'u' {
                u32::from_str_radix(digit, radix).map(Integer::U)
            } else {
                i64::from_str_radix(digit, radix).map(Integer::Abstract)
            }
            .ok()?,
        ))
    }
}
