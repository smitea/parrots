/// Splits text into clauses on Chinese/English sentence punctuation, keeping
/// punctuation; empty and punctuation-only segments are dropped; brackets,
/// quotes and other attach characters never form a segment on their own
pub fn split_clauses(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if is_clause_punct(ch) {
            flush_clause(&mut out, &mut cur);
        }
    }
    flush_clause(&mut out, &mut cur);
    out
}

fn flush_clause(out: &mut Vec<String>, cur: &mut String) {
    let t = cur.trim();
    if !t.is_empty() && t.chars().any(|c| !is_clause_punct(c) && !is_attach(c)) {
        out.push(t.to_string());
    }
    cur.clear();
}

/// Chinese/English sentence punctuation set; includes ASCII ','/'!' etc.:
/// opus-mt Chinese output uses half-width punctuation
// Known tradeoff: splitting on '.' and ',' breaks decimals (3.14) and
// thousands separators (1,000); harmless for TTS
pub(crate) fn is_clause_punct(ch: char) -> bool {
    matches!(
        ch,
        '。' | '！' | '？' | '；' | '，' | ',' | '!' | '?' | ';' | '.'
    )
}

/// Attach set: brackets/quotes/ellipses etc. only travel with a clause, never
/// trigger a split, and do not count as body content
pub(crate) fn is_attach(ch: char) -> bool {
    matches!(
        ch,
        '(' | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '（'
            | '）'
            | '《'
            | '》'
            | '〈'
            | '〉'
            | '「'
            | '」'
            | '『'
            | '』'
            | '【'
            | '】'
            | '〔'
            | '〕'
            | '"'
            | '\''
            | '‘'
            | '’'
            | '“'
            | '”'
            | '…'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_zh_keeps_punctuation() {
        assert_eq!(
            split_clauses("你好!今天天气很好。再见"),
            vec!["你好!", "今天天气很好。", "再见"]
        );
    }

    #[test]
    fn splits_on_ascii_comma() {
        // opus-mt Chinese output uses half-width commas; must split on them too
        assert_eq!(
            split_clauses("大家好,谢谢参加,今天讨论路线图。"),
            vec!["大家好,", "谢谢参加,", "今天讨论路线图。"]
        );
    }

    #[test]
    fn splits_en() {
        assert_eq!(
            split_clauses("Hello! How are you? Fine."),
            vec!["Hello!", "How are you?", "Fine."]
        );
    }

    #[test]
    fn empty_and_punct_only() {
        assert!(split_clauses("").is_empty());
        assert!(split_clauses("。!").is_empty());
    }

    #[test]
    fn closers_attach_not_split() {
        // Trailing ')' belongs to the attach set: no split, and attach-only
        // segments are dropped so orphans never reach TTS
        assert_eq!(
            split_clauses("你好!你好吗?(Hello! How are you?)"),
            vec!["你好!", "你好吗?", "(Hello!", "How are you?"]
        );
        // Lone closing bracket = attach-only segment; same semantics as
        // empty_and_punct_only, output is empty
        assert!(split_clauses(")").is_empty());
        assert!(split_clauses("。!)」…").is_empty());
    }
}
