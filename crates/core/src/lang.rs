use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Lang {
    Zh,
    En,
}

impl Lang {
    pub fn code(self) -> &'static str {
        match self {
            Lang::Zh => "zh",
            Lang::En => "en",
        }
    }

    pub fn parse(code: &str) -> Option<Lang> {
        match code.to_ascii_lowercase().as_str() {
            "zh" | "zh-cn" | "chinese" => Some(Lang::Zh),
            "en" | "en-us" | "english" => Some(Lang::En),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_common_codes() {
        assert_eq!(Lang::parse("zh"), Some(Lang::Zh));
        assert_eq!(Lang::parse("EN"), Some(Lang::En));
        assert_eq!(Lang::parse("zh-CN"), Some(Lang::Zh));
        assert_eq!(Lang::parse("fr"), None);
    }

    #[test]
    fn code_roundtrip() {
        for l in [Lang::Zh, Lang::En] {
            assert_eq!(Lang::parse(l.code()), Some(l));
        }
    }
}
