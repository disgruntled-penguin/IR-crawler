"""Rule-based transliteration of Indic scripts to Latin, and a phonetic key shared by both sides.

Unicode lays out every Indic block on the ISCII grid: the same offset in the Devanagari, Bengali,
Tamil or Telugu block is the same letter. One table on offsets (codepoint & 0x7F) therefore covers
all nine scripts. A name transliterated this way rarely matches the English spelling exactly
('albaanez' for 'Albanese'), so both sides are reduced to a phonetic key before matching.
"""
import re

from . import text_indic as ti

CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh",
    0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d",
    0x27: "dh", 0x28: "n", 0x29: "n", 0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y",
    0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh", 0x38: "s",
    0x39: "h",
}
# A nukta below a consonant marks a borrowed sound: क़ q, ख़ kh, ग़ g, ज़ z, ड़ r, ढ़ rh, फ़ f, य़ y.
NUKTA = {0x15: "q", 0x17: "g", 0x1C: "z", 0x21: "r", 0x22: "rh", 0x2B: "f", 0x2F: "y"}
VOWELS = {
    0x05: "a", 0x06: "aa", 0x07: "i", 0x08: "ii", 0x09: "u", 0x0A: "uu", 0x0B: "ri", 0x0C: "li", 0x0D: "e",
    0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au", 0x60: "rri",
}
SIGNS = {
    0x3E: "aa", 0x3F: "i", 0x40: "ii", 0x41: "u", 0x42: "uu", 0x43: "ri", 0x44: "rii", 0x45: "e", 0x46: "e",
    0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au", 0x56: "ai", 0x57: "au", 0x62: "li",
}
CANDRABINDU, ANUSVARA, VISARGA, NUKTA_SIGN, VIRAMA = 0x01, 0x02, 0x03, 0x3C, 0x4D
# Malayalam chillu letters and Bengali khanda ta: a final consonant with no vowel.
CHILLU = {0x0D7A: "n", 0x0D7B: "n", 0x0D7C: "r", 0x0D7D: "l", 0x0D7E: "l", 0x0D7F: "k", 0x09CE: "t"}
LABIALS = ("p", "b", "m", "f")


def _units(word):
    """Split a word into syllable units: [consonant, vowel] where vowel None means the inherent schwa,
    '' means killed by virama, and units with consonant None are independent vowels or nasals."""
    units = []
    for ch in word:
        o = ord(ch)
        if o in CHILLU:
            units.append([CHILLU[o], ""])
            continue
        if not 0x0900 <= o < 0x0D80:
            continue
        off = o & 0x7F
        if off in CONSONANTS:
            units.append([CONSONANTS[off], None, off])
        elif off == NUKTA_SIGN and units and len(units[-1]) == 3:
            units[-1][0] = NUKTA.get(units[-1][2], units[-1][0])
        elif off in SIGNS and units and units[-1][1] is None:
            units[-1][1] = SIGNS[off]
        elif off == VIRAMA and units and units[-1][1] is None:
            units[-1][1] = ""
        elif off in VOWELS:
            units.append([None, VOWELS[off]])
        elif off in (CANDRABINDU, ANUSVARA):
            units.append([None, "~"])
        elif off == VISARGA:
            units.append([None, "h"])
    return units


def _schwa_deletion(units):
    """Drop the inherent vowel at the word end (unless it closes a cluster, as in 'chandra') and in the
    medial context vowel + C_ + CV, as in कमला -> kamlaa."""
    cons = [i for i, u in enumerate(units) if u[0] is not None and u[1] is None]
    if not cons:
        return
    last = len(units) - 1
    if units[last][0] is not None and units[last][1] is None:
        cluster = last > 0 and units[last - 1][0] is not None and units[last - 1][1] == ""
        if not cluster and last > 0:
            units[last][1] = ""
    for i in range(len(units) - 2, 0, -1):
        u = units[i]
        if u[0] is None or u[1] is not None:
            continue
        prev, nxt = units[i - 1], units[i + 1]
        prev_vowel = prev[1] not in ("", None) or prev[1] is None and prev[0] is not None
        next_cv = nxt[0] is not None and nxt[1] != ""
        if prev_vowel and next_cv:
            u[1] = ""


def word(w):
    """Transliterate one Indic word to lowercase Latin."""
    units = _units(w)
    _schwa_deletion(units)
    out = []
    for i, u in enumerate(units):
        c, v = u[0], u[1]
        if c is None and v == "~":
            nxt = next((x[0] for x in units[i + 1:] if x[0] is not None), None)
            out.append("m" if nxt and nxt.startswith(LABIALS) else "n")
            continue
        if c is not None:
            out.append(c)
        out.append("a" if v is None else v)
    return "".join(out)


def translit(text):
    """Transliterate every Indic word in a text; Latin and digits pass through."""
    return " ".join(word(t) if ti.is_indic(t) else t for t in ti.tokens(text))


DIGRAPHS = [("dge", "je"), ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "g"), ("th", "t"), ("dh", "d"), ("bh", "b"),
            ("jh", "j"), ("zh", "j"), ("rh", "r"), ("ck", "k"), ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "s")]
VOICING = str.maketrans({"b": "p", "d": "t", "g": "k", "j": "c"})
SOFT_C = re.compile(r"c(?=[eiy])")


def key(s, fold_voicing=False):
    """Phonetic key used on both sides: aspirates and spelling variants folded, doubles collapsed, vowels
    and glides dropped after the first letter, any initial vowel (or y + vowel, as in यूक्रेन) class 'a'.

    fold_voicing also merges b/p, d/t, g/k, j/c, for Tamil, which does not write voicing; it is off
    otherwise because it merges Biden with Putin.
    """
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    # 'ch' is the affricate on both sides; any other English 'c' is s before e/i/y, else k.
    s = s.replace("chh", "\0").replace("ch", "\0")
    s = SOFT_C.sub("s", s).replace("c", "k").replace("\0", "c")
    for a, b in DIGRAPHS:
        s = s.replace(a, b)
    if fold_voicing:
        s = s.translate(VOICING)
    if len(s) > 1 and s[0] == "y" and s[1] in "aeiou":
        s = s[1:]
    head = "a" if s[0] in "aeiou" else s[0]
    rest = re.sub(r"[aeiouhy]", "", s[1:])
    out = head
    for ch in rest:
        if ch != out[-1]:
            out += ch
    return out


def name_key(token, fold_voicing=None):
    """Key of one token from either side: Indic words are transliterated first. Tamil words fold voicing
    unless told otherwise."""
    if ti.is_indic(token):
        if fold_voicing is None:
            fold_voicing = 0x0B80 <= ord(token[0]) < 0x0C00
        return key(word(token), fold_voicing)
    return key(token, bool(fold_voicing))
