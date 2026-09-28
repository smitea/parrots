//! Hotword corrector (Plan 6): rewrites homophone/near-homophone misspellings
//! in ASR output to the canonical term-list spelling.
//!
//! Two layers, with `#` comments and blank lines skipped:
//! ① Exact substring hits (the text already contains a term-list word,
//!    claimed as a placeholder so shorter hotwords' pinyin matching cannot
//!    swallow it);
//! ② Fuzzy pinyin: the text and each hotword are converted to toneless
//!    pinyin sequences and matched with a sliding window
//!    (same sounds with different tones also count: `稀有記` xī yǒu jì →
//!    `西游记` xī yóu jì);
//!    longest hotword wins, replaced with the term-list spelling.
//!
//! Non-Han characters never participate in pinyin matching (naturally
//! blocking false matches across punctuation);
//! hotwords containing non-Han characters only go through the exact path.

use std::collections::HashSet;
use std::path::Path;

/// A single candidate match: text char range [start, start+len) + replacement term.
struct Candidate<'a> {
    start: usize,
    len: usize,
    term: &'a str,
}

/// Hotword term list and corrector.
#[derive(Debug, Default, Clone)]
pub struct Hotwords {
    /// Raw term list (longest first, for longest-first matching)
    terms: Vec<String>,
    /// Toneless pinyin sequences of pure-Han hotwords (indexed like terms;
    /// None for non-pure-Han terms)
    pinyin_seqs: Vec<Option<Vec<String>>>,
}

impl Hotwords {
    /// Loads from a term-list file (UTF-8, one term per line; `#` comments
    /// and blank lines skipped).
    /// Missing file → empty term list (no error; feature silently off).
    pub fn load(path: &Path) -> Self {
        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => return Self::default(),
        };
        Self::from_text(&content)
    }

    /// Builds from a term-list string (parsing shared by tests and load).
    pub fn from_text(text: &str) -> Self {
        let mut terms: Vec<String> = text
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| l.to_string())
            .collect();
        // Longest first: original order preserved for equal lengths
        terms.sort_by_key(|t| std::cmp::Reverse(t.chars().count()));
        let pinyin_seqs = terms.iter().map(|t| han_pinyin_seq(t)).collect();
        Self { terms, pinyin_seqs }
    }

    /// Builds from a term iterator.
    pub fn from_terms<'a>(terms: impl IntoIterator<Item = &'a str>) -> Self {
        Self::from_text(&terms.into_iter().collect::<Vec<_>>().join("\n"))
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Joins the term list (for whisper initial_prompt injection).
    pub fn prompt_text(&self) -> String {
        self.terms.join(" ")
    }

    /// Read-only view of the term list (for TextPolisher context injection, Plan 7).
    pub fn terms(&self) -> &[String] {
        &self.terms
    }

    /// Corrects text: exact-hit placeholders + longest-first fuzzy pinyin replacement.
    pub fn correct(&self, text: &str) -> String {
        if self.is_empty() {
            return text.to_string();
        }
        let chars: Vec<char> = text.chars().collect();
        let text_pinyin: Vec<Option<String>> = chars.iter().map(han_pinyin).collect();

        // Candidates: ① exact hits; ② fuzzy pinyin (pure-Han hotwords only)
        let mut candidates: Vec<Candidate> = Vec::new();
        for (ti, term) in self.terms.iter().enumerate() {
            let tlen = term.chars().count();
            if tlen == 0 || tlen > chars.len() {
                continue;
            }
            let term_chars: Vec<char> = term.chars().collect();
            for start in 0..=(chars.len() - tlen) {
                // ① Exact substring hit
                if chars[start..start + tlen] == term_chars[..] {
                    candidates.push(Candidate {
                        start,
                        len: tlen,
                        term,
                    });
                    continue;
                }
                // ② Fuzzy pinyin: pure-Han hotword + all-Han text range + same sequence
                if let Some(seq) = &self.pinyin_seqs[ti] {
                    let seq_same_len = &text_pinyin[start..start + tlen];
                    if seq.len() == tlen
                        && seq_same_len
                            .iter()
                            .zip(seq)
                            .all(|(tp, sp)| tp.as_deref() == Some(sp.as_str()))
                    {
                        candidates.push(Candidate {
                            start,
                            len: tlen,
                            term,
                        });
                    }
                }
            }
        }

        // Disambiguation: longest first, leftmost on ties; greedily pick
        // non-overlapping candidates
        candidates.sort_by_key(|c| (std::cmp::Reverse(c.len), c.start));
        let mut selected: Vec<Candidate> = Vec::new();
        let mut taken: HashSet<usize> = HashSet::new();
        for cand in candidates {
            let span = cand.start..cand.start + cand.len;
            if span.clone().any(|i| taken.contains(&i)) {
                continue;
            }
            taken.extend(span);
            selected.push(cand);
        }
        selected.sort_by_key(|c| c.start);

        // Rebuild: matched ranges replaced with the term-list spelling, rest unchanged
        let mut out = String::with_capacity(text.len());
        let mut pos = 0usize;
        for cand in selected {
            while pos < cand.start {
                out.push(chars[pos]);
                pos += 1;
            }
            out.push_str(cand.term);
            pos += cand.len;
        }
        out.extend(chars[pos..].iter());
        out
    }
}

/// Toneless pinyin of a single Han character (None for non-Han).
fn han_pinyin(c: &char) -> Option<String> {
    use pinyin::ToPinyin;
    c.to_pinyin().map(|p| p.plain().to_string())
}

/// Pinyin sequence of a string; any non-Han character → None (excluded from fuzzy matching).
fn han_pinyin_seq(s: &str) -> Option<Vec<String>> {
    let mut seq = Vec::new();
    for c in s.chars() {
        seq.push(han_pinyin(&c)?);
    }
    Some(seq)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homophone_traditional_is_corrected() {
        let hw = Hotwords::from_terms(["西游记"]);
        // 稀有記 (xī yǒu jì, traditional 記) and 西游记 (xī yóu jì) share the
        // same sounds with different tones
        assert_eq!(hw.correct("我再看稀有記"), "我再看西游记");
    }

    #[test]
    fn exact_hit_is_untouched() {
        let hw = Hotwords::from_terms(["西游记"]);
        assert_eq!(hw.correct("我再看西游记"), "我再看西游记");
    }

    #[test]
    fn longest_term_wins() {
        let hw = Hotwords::from_terms(["西游", "西游记"]);
        assert_eq!(
            hw.correct("稀有記"),
            "西游记",
            "longer 西游记 should win over 西游"
        );
    }

    #[test]
    fn no_match_returns_as_is() {
        let hw = Hotwords::from_terms(["西游记", "红楼梦"]);
        let text = "今天天气不错,适合出门散步";
        assert_eq!(hw.correct(text), text);
    }

    #[test]
    fn multiple_hotwords_corrected() {
        let hw = Hotwords::from_terms(["西游记", "红楼梦"]);
        assert_eq!(hw.correct("稀有記与红楼夢"), "西游记与红楼梦");
    }

    #[test]
    fn match_does_not_cross_punctuation() {
        let hw = Hotwords::from_terms(["西游记"]);
        // 稀有 and 記 are separated by punctuation, so the syllables are not
        // a contiguous sequence and must not match
        assert_eq!(hw.correct("稀有、記事本"), "稀有、記事本");
    }

    #[test]
    fn comments_and_blank_lines_skipped() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("hotwords.txt");
        std::fs::write(&path, "# 专有名词\n西游记\n\n红楼梦\n").unwrap();
        let hw = Hotwords::load(&path);
        assert_eq!(hw.len(), 2);
        assert_eq!(hw.correct("稀有記"), "西游记");
    }

    #[test]
    fn missing_file_loads_empty() {
        let hw = Hotwords::load(Path::new("/nonexistent/hotwords.txt"));
        assert!(hw.is_empty());
        assert_eq!(hw.correct("任意文本"), "任意文本");
    }

    #[test]
    fn non_han_hotword_only_exact_path() {
        let hw = Hotwords::from_terms(["iPhone"]);
        assert_eq!(hw.correct("我喜欢iPhone"), "我喜欢iPhone");
        // The pinyin path skips non-Han hotwords: nothing to mis-replace
        assert_eq!(hw.correct("我喜欢爱疯"), "我喜欢爱疯");
    }

    #[test]
    fn prompt_text_joins_terms() {
        let hw = Hotwords::from_terms(["西游记", "红楼梦"]);
        assert_eq!(hw.prompt_text(), "西游记 红楼梦");
    }
}
