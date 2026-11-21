"""Text pipeline for Indic-script suspects: normalisation, tokens, sentences, and the anchors that
survive translation (numbers, dates, Latin-script tokens).

text.fold strips combining marks, which would delete every Devanagari vowel sign, so Indic text
gets NFC here instead. Numbers are canonicalised on both sides: '3.5 करोड़' and '35 million' both
become 35000000, so a Hindi article and its English source meet on the same number.
"""
import re
import unicodedata

from . import text as tx

BLOCKS = tuple(tx.INDIC_BLOCKS)
DIGITS = {base + 0x66 + i: str(i) for base in BLOCKS for i in range(10)}
DIGITS.update({0x200C: None, 0x200D: None, 0x00A0: " "})

INDIC_CHAR = "ऀ-ॣ०-ൿ"
TOKEN = re.compile(rf"\d+(?:[.,]\d+)*|[A-Za-z]+(?:['\-][A-Za-z0-9]+)*|[{INDIC_CHAR}]+")
LATIN_SPAN = re.compile(r"[A-Za-z][A-Za-z0-9.\-']*(?:\s+[A-Za-z0-9][A-Za-z0-9.\-']*)*")
SENT_SPLIT = re.compile(r"(?<=[।॥.?!])\s+|\n+")
QUOTE = re.compile(r'"([^"]{15,400})"')

HI_STOPWORDS = frozenset("""
का के की को में से पर ने है हैं था थे थी थीं हो होगा होगी होंगे हुआ हुए हुई और या एक यह ये वह वे इस इन उस उन
कि जो तो भी ही तक साथ लिए लिये बाद पहले द्वारा कहा कहना कर करने करते करता करती किया किए किये गया गए गई
रहा रहे रही अपने अपनी अपना इसके उसके उनके इसकी उसकी उनकी जब तब अब कोई कुछ सभी सब बहुत अधिक कम नहीं न
वाले वाली वाला होने कई लगभग अन्य जा जाता जाती जाते जाएगा सकता सकते सकती गयी दिया दी दिए लेकिन परंतु
किसी जिसमें जिसके जिससे जिन्होंने उन्होंने इसमें उसमें इसे उसे उन्हें इन्हें आज कल प्रति बीच दौरान ओर तरह वर्ष
साल दिन लोग लोगों बताया अनुसार रूप वजह कारण मामले मामला बात एवं तथा यदि क्योंकि ऐसे ऐसा वहां यहां जहां
""".split())

MULTIPLIERS = {
    "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "crore": 1e7, "crores": 1e7, "million": 1e6, "millions": 1e6,
    "mn": 1e6, "billion": 1e9, "billions": 1e9, "bn": 1e9, "trillion": 1e12,
    "हज़ार": 1e3, "हजार": 1e3, "लाख": 1e5, "करोड़": 1e7, "करोड": 1e7, "अरब": 1e9, "खरब": 1e11,
    "मिलियन": 1e6, "बिलियन": 1e9, "ट्रिलियन": 1e12,
    "ஆயிர": 1e3, "லட்ச": 1e5, "கோடி": 1e7, "மில்லியன்": 1e6, "பில்லியன்": 1e9,
    "వేల": 1e3, "వెయ్య": 1e3, "లక్ష": 1e5, "కోట": 1e7, "మిలియన్": 1e6, "బిలియన్": 1e9,
    "হাজার": 1e3, "লাখ": 1e5, "লক্ষ": 1e5, "কোটি": 1e7, "মিলিয়ন": 1e6, "বিলিয়ন": 1e9,
}
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100, "half": 0.5,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "बीस": 20, "तीस": 30, "पचास": 50, "सौ": 100, "डेढ़": 1.5, "ढाई": 2.5,
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
MONTH_WORDS = {m: i + 1 for i, m in enumerate(MONTHS)}
MONTH_WORDS.update({m[:3]: i + 1 for i, m in enumerate(MONTHS)})
MONTH_WORDS.update({"sept": 9})
for i, names in enumerate([
        "जनवरी", "फ़रवरी फरवरी", "मार्च", "अप्रैल अप्रेल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर सितम्बर",
        "अक्टूबर अक्तूबर", "नवंबर नवम्बर", "दिसंबर दिसम्बर"]):
    for n in names.split():
        MONTH_WORDS[n] = i + 1
WEEKDAYS = {"रविवार": "sunday", "सोमवार": "monday", "मंगलवार": "tuesday", "बुधवार": "wednesday",
            "गुरुवार": "thursday", "बृहस्पतिवार": "thursday", "शुक्रवार": "friday", "शनिवार": "saturday"}


def normalize(text):
    """NFC (nukta letters decompose to base + nukta, as the composition exclusions require), Indic digits to
    ASCII, joiners dropped, curly quotes straightened. Combining vowel signs are kept."""
    return unicodedata.normalize("NFC", text).translate(DIGITS).translate(tx.QUOTE_MAP)


def _norm_keys(d):
    return {normalize(k): v for k, v in d.items()}


MULTIPLIERS = _norm_keys(MULTIPLIERS)
NUMBER_WORDS = _norm_keys(NUMBER_WORDS)
MONTH_WORDS = _norm_keys(MONTH_WORDS)
WEEKDAYS = _norm_keys(WEEKDAYS)
HI_STOPWORDS = frozenset(normalize(w) for w in HI_STOPWORDS)


def is_indic(tok):
    return bool(tok) and 0x0900 <= ord(tok[0]) < 0x0D80


def tokens(text):
    """Lowercased Latin words, numbers (commas dropped, decimals kept) and whole Indic words."""
    out = []
    for t in TOKEN.findall(normalize(text)):
        if t[0].isdigit():
            out.append(t.replace(",", ""))
        else:
            out.append(t.lower())
    return out


def content_tokens(toks):
    return [t for t in toks if t not in HI_STOPWORDS and t not in tx.STOPWORDS]


def sentences(text, min_tokens=3):
    """Split on the danda and double danda as well as Latin sentence ends and line breaks."""
    text = normalize(text)
    out = [s.strip() for s in SENT_SPLIT.split(text)]
    return [s for s in out if len(TOKEN.findall(s)) >= min_tokens]


def quotes(text, min_words=4):
    return [q.strip() for q in QUOTE.findall(normalize(text)) if len(q.split()) >= min_words]


def _multiplier(tok):
    if tok in MULTIPLIERS:
        return MULTIPLIERS[tok]
    if is_indic(tok):
        # Dravidian case endings attach to the word ('லட்சத்து', 'కోట్ల'), so match by stem.
        for stem, v in MULTIPLIERS.items():
            if is_indic(stem) and len(stem) >= 3 and tok.startswith(stem):
                return v
    return None


def canon(value):
    """One string per numeric value: integers without decimals, others to at most 4 places."""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4f}".rstrip("0").rstrip(".")


def numbers(text, toks=None):
    """Canonical numbers: digits or a number word, scaled by a following multiplier word.

    Bare number words are skipped ('एक' is also the article 'a'); single digits are kept only when
    scaled, since alone they are everywhere.
    """
    toks = toks if toks is not None else tokens(text)
    out = []
    for i, t in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        mult = _multiplier(nxt)
        if t[0].isdigit():
            try:
                v = float(t)
            except ValueError:
                continue
            if v == 0:
                continue
            if mult:
                out.append(canon(v * mult))
            elif len(t.replace(".", "")) >= 2:
                out.append(canon(v))
        elif t in NUMBER_WORDS and mult:
            out.append(canon(NUMBER_WORDS[t] * mult))
    return out


def month_of(tok):
    return MONTH_WORDS.get(tok)


def dates(text, toks=None):
    """(month, day) pairs from 'day month' or 'month day', as in '26 जुलाई' or 'July 26'."""
    toks = toks if toks is not None else tokens(text)
    out = []
    for i, t in enumerate(toks):
        m = month_of(t)
        if not m:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(toks) and toks[j].isdigit() and 1 <= int(toks[j]) <= 31 and len(toks[j]) <= 2:
                out.append((m, int(toks[j])))
                break
    return out


def latin_tokens(text):
    """Latin-script tokens inside an Indic text: brands, acronyms, names left untranslated."""
    toks = []
    for span in LATIN_SPAN.findall(normalize(text)):
        toks += [t for t in tx.tokens(span) if not t[0].isdigit()]
    return [t for t in toks if t not in tx.STOPWORDS and len(t) > 1]


def weekday_terms(toks):
    return [WEEKDAYS[t] for t in toks if t in WEEKDAYS]
