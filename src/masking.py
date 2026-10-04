"""Masking primitives for the private ingest of the real ATS data.

Everything here is pure and unit-tested. The principle throughout is fail-closed: if
something can't be masked, it raises, and the ingest writes nothing.

Layers applied to free text, in order:
1. labeled personal-data lines ("Estado civil: ...", "Bairro: ...")
2. patterns (email, URL, ID numbers, postcode, dates, phones, long digit runs, marital
   status, number of children)
3. the applicant's own name, taken from the structured record (accent-insensitive)
4. name chains: a common first name followed by name-like tokens, from a dictionary
5. named-entity recognition for people, only for MIXED-CASE short fields

Why layer 4 and not NER for CVs: the real CV text is 100% lowercase, and cased NER models
find names mostly through capitalization. On lowercase text the NER we measured recalled
~14% of names, the dictionary chain ~90% and ran ~15x faster (see docs/masking.md).
"""

import csv
import hashlib
import hmac
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

# --- accent folding -----------------------------------------------------------------------


def _fold_char(char: str) -> str:
    decomposed = unicodedata.normalize("NFD", char)
    return (decomposed[0] if decomposed else char).lower()[:1] or char


def fold(text: str) -> str:
    """Lowercase and strip accents while keeping the SAME LENGTH, so character offsets
    found on the folded text are valid on the original."""
    return "".join(_fold_char(c) for c in text)


# --- layers 1 and 2: labeled lines and patterns -------------------------------------------

_YEAR_RANGE = re.compile(r"(?:19|20)\d{2}\s*[.\-–]?\s*(?:19|20)\d{2}")
# looser: also a year pair split by a newline, which the independent estimate must not call a phone
_YEAR_PAIR_LOOSE = re.compile(r"(?:19|20)\d{2}\s*[.-]?\s*(?:19|20)\d{2}")

# Phone numbers are matched by several explicit shapes instead of one permissive regex,
# because a permissive one also eats year ranges such as "2015-2018".
_PSEP = r"[ \t.\-–]{0,3}"  # " - ", "-", " – ", "." between the digit groups
_PHONE = re.compile(
    rf"(?:\+?55{_PSEP})?\(\s?\d{{2}}\s?\){_PSEP}9?{_PSEP}\d{{4}}{_PSEP}\d{{4}}"  # (11) 98765-4321
    rf"|\+55{_PSEP}\d{{2}}{_PSEP}9?{_PSEP}\d{{4}}{_PSEP}\d{{4}}"  # +55 11 98765 4321
    r"|(?<![\d-])\d{2}[ \t.-]9?\d{4}[ \t.\-–]{1,3}\d{4}(?![\d-])"  # 11 98765-4321
    r"|(?<![\d-])9?\d{4}[ \t]?[-–][ \t]?\d{4}(?![\d-])"  # 98765-4321, 3333 - 4444
    r"|(?<![\d-])\d{2}\)\s?9?[ \t.\-–]?\d{4}[ \t.\-–]{1,3}\d{4}(?![\d-])"  # 11) 98765-4321
    r"|(?<![\d-])9\d{4}[ .]\d{4}(?![\d-])"  # 98765 4321 (mobile, spaced)
    r"|(?<![\d-])[2-5]\d{3}[ .]\d{4}(?![\d-])"  # 3333 4444 (landline, spaced)
)

# CVs often carry labeled personal-data lines. The value after such a label is redacted,
# whatever it looks like: patterns can't recognise a neighborhood or "12 de março de 1990".
# Address-like labels take the rest of the line; short attributes stop at the next separator
# so "Idade: 32, Solteiro" keeps what follows.
_LINE_LABELS = (
    r"endere[cç]o|bairro|cidade|cep|telefone|celular|tel|fone|e-?mail|contato"
    r"|nome(?: completo)?|filia[cç][aã]o|pai|m[aã]e|skype"
)
_SHORT_LABELS = (
    r"data de nascimento|nascimento|idade|estado civil|naturalidade|nacionalidade|sexo"
    r"|g[eê]nero|ra[cç]a|cor|rg|cpf|cnh|filhos|dependentes"
)
_SEP = r"[ \t]*[:\-–=][ \t]*"
_LABELED = [
    # label (with or without a separator) alone on a line, value on the next line
    re.compile(
        rf"(?<!\w)((?:{_LINE_LABELS}|{_SHORT_LABELS})(?:{_SEP})?[ \t]*\n[ \t]*)([^\n]+)",
        re.IGNORECASE,
    ),
    re.compile(rf"(?<!\w)((?:{_LINE_LABELS}){_SEP})([^\n]*)", re.IGNORECASE),
    re.compile(rf"(?<!\w)((?:{_SHORT_LABELS}){_SEP})([^,;|\n]*)", re.IGNORECASE),
    # a short label followed directly by a number: "idade 32", "rg 12 345 678 9"
    re.compile(
        r"(?<!\w)((?:idade|nascimento|data de nascimento|rg|cpf|cnh)[ \t]+)(\d[\d./ \-]*)",
        re.IGNORECASE,
    ),
]
_REDACTED = "[REDACTED]"
_PT_MONTHS = (
    "janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    (
        "URL",
        re.compile(
            r"(?:https?://|www\.)\S+"
            r"|\b(?:linkedin|facebook|instagram|twitter|github)\.com\S*",
            re.IGNORECASE,
        ),
    ),
    (
        "ID",  # CPF, CNPJ, RG (dotted or spaced)
        re.compile(
            r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"
            r"|\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"
            r"|\b\d{1,2}[ .]\d{3}[ .]\d{3}[ .-][\dxX]\b"
        ),
    ),
    ("POSTCODE", re.compile(r"\b\d{5}-\d{3}\b")),
    # numeric, full; digit lookarounds (not \b) so a date glued to a letter is still caught
    ("DATE", re.compile(r"(?<!\d)\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)?\d{2}(?!\d)")),
    ("DATE", re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")),  # ISO
    ("DATE", re.compile(rf"\b\d{{1,2}} de (?:{_PT_MONTHS}) de (?:19|20)\d{{2}}\b", re.IGNORECASE)),
    ("AGE", re.compile(r"\b\d{2} anos de idade\b", re.IGNORECASE)),
    (
        "ATTRIBUTE",  # marital status and number of children
        re.compile(
            r"\b(?:solteir[oa]s?|casad[oa]s?|divorciad[oa]s?|vi[uú]v[oa]s?"
            r"|separad[oa]s? judicialmente|uni[aã]o est[aá]vel"
            r"|(?:sem|com|\d{1,2}|um|uma|dois|duas|tr[eê]s) filh[oa]s?)\b",
            re.IGNORECASE,
        ),
    ),
    ("PHONE", _PHONE),
    ("NUMBER", re.compile(r"(?<!\d)\d{9,}(?!\d)")),
    (
        "ADDRESS",  # street + optional number; city and state are kept
        re.compile(
            r"(?:\b(?:rua|avenida|alameda|travessa|rodovia|estrada|pra[cç]a|r\.|av\.)(?:\s*:\s*|\s+)"
            r"|(?<=anos)r\.\s*:\s*)"  # "34 anosr.: ..." (glued by text extraction)
            r"[a-zà-ú0-9 .'\[\]-]{2,40}?(?:,?\s*(?:n[º°o.]?\s*)?\d{1,5})?"
            r"(?=[,;\n]|\s[-–]\s|\.\s|$)",
            re.IGNORECASE,
        ),
    ),
    (
        "LOCATION",  # neighborhood-level names; "parque tecnológico" is not one
        re.compile(
            r"\b(?:jardim|vila|residencial|bairro|loteamento|recanto|ch[aá]cara)\s+"
            r"(?:(?:s[aã]o|santa|santo)\s+[a-zà-ú]{3,}|[a-zà-ú]{3,})"
            r"(?:\s+(?:de|da|do|dos|das)\s+[a-zà-ú]{3,})?",
            re.IGNORECASE,
        ),
    ),
]
_BARE_AGE = re.compile(r"(?<![\w\[])(?:1[89]|[2-5]\d|6[0-5]) anos(?!\w\w)")  # tolerates "anosr."
_REPLACEMENT = {"ATTRIBUTE": _REDACTED}


def _redact_labeled(text: str, stats: Counter | None = None) -> str:
    for pattern in _LABELED:

        def repl(match: re.Match[str]) -> str:
            if match.group(2).strip() in ("", _REDACTED):
                return match.group()
            if stats is not None:
                stats["LABELED"] += 1
            return f"{match.group(1)}{_REDACTED}"

        text = pattern.sub(repl, text)
    return text


def mask_patterns(text: str, stats: Counter | None = None) -> str:
    """Redact labeled personal-data lines, then emails, URLs, IDs, dates, phones, etc."""
    text = _redact_labeled(text, stats)
    for label, pattern in _PATTERNS:

        def repl(match: re.Match[str], label: str = label) -> str:
            if label == "PHONE" and _YEAR_RANGE.fullmatch(match.group().strip(" ()")):
                return match.group()  # a year range, not a phone number
            if stats is not None:
                stats[label] += 1
            return _REPLACEMENT.get(label, f"[{label}]")

        text = pattern.sub(repl, text)
    return _mask_bare_ages(text, stats)


def _mask_bare_ages(text: str, stats: Counter | None = None) -> str:
    """ "34 anos" is an age only in a personal-data context: after a comma, bracket or line start
    and before punctuation or a line end ("nascida em [DATE], 34 anos."). "mais de 10 anos de
    experiência" and "com 15 anos de ..." are left alone."""

    def repl(match: re.Match[str]) -> str:
        before = text[max(0, match.start() - 14) : match.start()].rstrip(" \t")
        after = text[match.end() : match.end() + 14].lstrip(" \t")
        keyword = re.search(r"\b(?:tenho|idade)$", before)
        personal = re.search(r"(?:\[[a-z_]+\]|brasileir[oa]),$", before, re.I)
        left = not before or before[-1] in ",;(-–|/]:\n"
        right = not after or after[0] in ",.;)|/\n-–[("
        if not (keyword or personal or (left and right)):
            return match.group()
        if stats is not None:
            stats["AGE"] += 1
        return "[AGE]"

    return _BARE_AGE.sub(repl, text)


# --- independent residual check -----------------------------------------------------------
# Deliberately NOT built from the masking patterns above: re-using them would only prove the
# masker agrees with itself. These are looser, separately written shapes. They estimate what
# may have slipped through; they are heuristics, not proof of anything.

_INDEPENDENT_CHECKS: dict[str, re.Pattern[str]] = {
    "email_or_at_sign": re.compile(r"@"),
    "url_or_domain": re.compile(r"(?:https?:|www\.|\.com\b|\.br\b)", re.IGNORECASE),
    "iso_date": re.compile(r"(?<!\d)(?:19|20)\d{2}-\d{1,2}-\d{1,2}(?!\d)"),
    "numeric_date": re.compile(r"(?<!\d)\d{1,2}[/.]\d{1,2}[/.]\d{2,4}(?!\d)"),
    "phone_like": re.compile(r"(?<!\d)(?:\(\d{2}\)|\d{2})?[\s-]?9?\d{4}[\s.-]\d{4}(?!\d)"),
    "long_digit_run": re.compile(r"\d{8,}"),
    "labeled_value": re.compile(
        rf"(?<!\w)(?:{_LINE_LABELS}|{_SHORT_LABELS})[ \t]*[:\-–=][ \t]*"
        r"(?!\[REDACTED\])[^\W_]{2,}",
        re.IGNORECASE,
    ),
    "label_then_digits": re.compile(
        r"(?<!\w)(?:idade|nascimento|rg|cpf|cnh)[ \t]+\d", re.IGNORECASE
    ),
    "label_alone_then_text": re.compile(
        r"(?im)^[ \t]*(?:nome(?: completo)?|endere[cç]o|telefone|celular|e-?mail)"
        r"[ \t]*\n[ \t]*(?!\[)\S"
    ),
    "age_like": re.compile(r"(?:^|[,;(\n])[ \t]*\d{2} anos[ \t]*(?:[,.;)\n]|$)", re.MULTILINE),
    "street_like": re.compile(r"\b(?:rua|avenida|alameda|travessa|rodovia)[ \t]+\w", re.IGNORECASE),
    "neighborhood_like": re.compile(
        r"\b(?:jardim|vila|bairro|residencial)[ \t]+[^\W\d_]{3}", re.IGNORECASE
    ),
    "marital_or_children": re.compile(
        r"\b(?:solteir|casad|divorciad|viuv|viúv)[oa]s?\b|\b\d+ filh[oa]s?\b"
        r"|\bsem filhos\b|\b(?:um|uma|dois|duas) filh[oa]s?\b",
        re.IGNORECASE,
    ),
}


def independent_residual_counts(texts: Sequence[str]) -> dict[str, int]:
    """Number of texts in which each independently-written check still fires."""
    counts = {name: 0 for name in _INDEPENDENT_CHECKS}
    for text in texts:
        for name, pattern in _INDEPENDENT_CHECKS.items():
            for match in pattern.finditer(text):
                if name == "phone_like" and _YEAR_PAIR_LOOSE.fullmatch(
                    match.group().strip(" ()\n\t")
                ):
                    continue
                counts[name] += 1
                break
    return counts


def residual_pattern_counts(texts: Sequence[str]) -> dict[str, int]:
    """How many of the (already masked) texts still match the masker's OWN patterns.
    This only proves self-consistency; see `independent_residual_counts` for the real check."""
    counts = {label: 0 for label, _ in _PATTERNS} | {"LABELED": 0}
    for text in texts:
        counts["LABELED"] += any(
            m.group(2).strip() not in ("", _REDACTED)
            for pattern in _LABELED
            for m in pattern.finditer(text)
        )
        for label, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                if label == "PHONE" and _YEAR_RANGE.fullmatch(match.group().strip(" ()")):
                    continue
                counts[label] += 1
                break
    return counts


# --- layer 3: the applicant's own name ----------------------------------------------------

_NAME_PARTICLES = {"da", "de", "do", "das", "dos", "e", "di", "du", "del", "van", "von"}
_WORD = re.compile(r"[^\W\d_]+")


def name_tokens(name: str | None) -> list[str]:
    """Distinctive folded tokens of a person's name (no particles, nothing under 3 letters)."""
    if not name:
        return []
    return [w for w in map(fold, _WORD.findall(name)) if len(w) >= 3 and w not in _NAME_PARTICLES]


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def mask_own_name(text: str, name: str | None, stats: Counter | None = None) -> str:
    """Mask the person's own name, accent- and case-insensitively.

    Distinctive tokens are masked individually; the whole name is also matched as a phrase,
    which catches short tokens ("Li Wu") that the token rule skips.
    """
    if not name or not name.strip():
        return text
    folded = fold(text)
    spans: list[tuple[int, int]] = []
    tokens = sorted(set(name_tokens(name)), key=len, reverse=True)
    if tokens:
        pattern = re.compile(rf"(?<!\w)(?:{'|'.join(map(re.escape, tokens))})(?!\w)")
        spans += [m.span() for m in pattern.finditer(folded)]
    words = [fold(w) for w in _WORD.findall(name)]
    if len(words) >= 2:
        phrase = re.compile(rf"(?<!\w){r'[\W_]+'.join(map(re.escape, words))}(?!\w)")
        spans += [m.span() for m in phrase.finditer(folded)]
    merged = _merge_spans(spans)
    for start, end in reversed(merged):
        text = text[:start] + "[NAME]" + text[end:]
    if stats is not None and merged:
        stats["OWN_NAME"] += len(merged)
    return text


# --- layer 4: name chains from a dictionary -----------------------------------------------

_SAINT = {"sao", "santo", "santa"}  # "são paulo" is a place, not a person
_PARTICLES = {"da", "de", "do", "das", "dos"}
_MAX_UNKNOWN = 2  # unknown (surname-like) words a chain may absorb
# Unknown surnames are only trusted near the top of a text, where a person's own name sits; deeper
# in a CV an unknown word after a first name is more often an employer or a school.
_HEADER_CHARS = 400
_ALWAYS_KNOWN = frozenset(
    "janeiro fevereiro marco abril maio junho julho agosto setembro outubro novembro dezembro "
    "ltda eireli cia companhia universidade faculdade instituto centro escola colegio".split()
)
_TOKEN = re.compile(r"[^\W\d_]+")
_GAP = re.compile(r"[ \t\-–]+")
# in the first lines a name is often extracted one word per line ("rhandy\nmendes\nferreira")
_GAP_HEADER = re.compile(r"[ \t\-–]*\n?[ \t\-–]*")
_HEADER_NEWLINE_CHARS = 150


@dataclass(frozen=True)
class NameDictionary:
    """first_names start a chain; name_tokens (first names + surnames) continue it."""

    first_names: frozenset[str]
    name_tokens: frozenset[str]
    # Ordinary vocabulary. When given, a word that is NOT in it may continue a chain as an
    # unknown surname ("marcela lucindo"); without it only known name tokens continue one.
    known_words: frozenset[str] = frozenset()

    @classmethod
    def build(
        cls,
        first_names: Iterable[str],
        extra_tokens: Iterable[str] = (),
        known_words: Iterable[str] = (),
    ) -> "NameDictionary":
        first = frozenset(t for t in map(fold, first_names) if len(t) >= 3)
        extra = frozenset(t for t in map(fold, extra_tokens) if len(t) >= 3)
        words = frozenset(t for t in map(fold, known_words) if len(t) >= 3)
        return cls(first, first | extra, words | _ALWAYS_KNOWN if words else words)


def load_ibge_first_names(path: Path, top_k: int = 3000) -> list[str]:
    """The top_k most common first names from the IBGE census file (columns: Nome, then one
    count per decade). The full list has ~64k entries, many of them ordinary words, so only
    the popular ones are used."""
    totals: list[tuple[int, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row or not row[0].strip():
                continue
            count = sum(int(c) for c in row[1:] if c.strip().isdigit())
            totals.append((count, row[0].strip()))
    totals.sort(reverse=True)
    return [name for _, name in totals[:top_k]]


def mask_name_chains(text: str, dictionary: NameDictionary, stats: Counter | None = None) -> str:
    """Mask runs of 2-4 name tokens that start with a common first name, on a single line.

    Works on lowercase text. Particles ("da", "dos") are allowed inside a chain; a chain never
    starts right after "são"/"santo"/"santa"; commas end a chain, so lists of first names are
    not glued into one person.
    """
    tokens = [(fold(m.group()), m.start(), m.end()) for m in _TOKEN.finditer(text)]
    spans: list[tuple[int, int]] = []
    i = 0
    while i < len(tokens):
        word = tokens[i][0]
        gap = (
            _GAP_HEADER if dictionary.known_words and tokens[i][1] < _HEADER_NEWLINE_CHARS else _GAP
        )
        # "santo andré", "santa catarina": saint words are IBGE first names but start places
        starts = (
            word in dictionary.first_names
            and word not in _SAINT
            and not (i > 0 and tokens[i - 1][0] in _SAINT)
        )
        # "rhandy mendes ferreira": a first name the list lacks, but two known surnames follow
        lead_unknown = (
            not starts
            and bool(dictionary.known_words)
            and tokens[i][1] < _HEADER_CHARS
            and len(word) >= 4
            and word not in dictionary.known_words
            and word not in dictionary.name_tokens
            and i + 2 < len(tokens)
            and all(
                tokens[k][0] in dictionary.name_tokens
                and tokens[k][0] not in _SAINT
                and gap.fullmatch(text[tokens[k - 1][2] : tokens[k][1]])
                for k in (i + 1, i + 2)
            )
        )
        if starts or lead_unknown:
            last, count, j, unknown = i, 1, i + 1, int(lead_unknown)
            while j < len(tokens) and count < 4:
                if not gap.fullmatch(text[tokens[j - 1][2] : tokens[j][1]]):
                    break
                nxt = tokens[j][0]
                if nxt in _PARTICLES and j + 1 < len(tokens):
                    after = tokens[j + 1][0]
                    gap2 = text[tokens[j][2] : tokens[j + 1][1]]
                    if after in dictionary.name_tokens and gap.fullmatch(gap2):
                        j += 1
                        continue
                if nxt in dictionary.name_tokens and len(nxt) >= 3:
                    last, count, j = j, count + 1, j + 1
                elif (
                    dictionary.known_words
                    and tokens[i][1] < _HEADER_CHARS
                    and unknown < _MAX_UNKNOWN
                    and len(nxt) >= 4
                    and nxt not in dictionary.known_words
                    and nxt not in _PARTICLES
                ):
                    last, count, j, unknown = j, count + 1, j + 1, unknown + 1
                else:
                    break
            if count >= 2:
                spans.append((tokens[i][1], tokens[last][2]))
                i = last + 1
                continue
        i += 1
    for start, end in reversed(spans):
        text = text[:start] + "[NAME]" + text[end:]
    if stats is not None and spans:
        stats["NAME_CHAIN"] += len(spans)
    return text


# --- layer 5: named-entity recognition (mixed-case fields only) ---------------------------


class EntityMasker(Protocol):
    counts: Counter

    def mask_many(self, texts: Sequence[str]) -> list[str]: ...


class NoEntityMasker:
    """Identity masker: used where NER is not wanted, and for fast runs and tests."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def mask_many(self, texts: Sequence[str]) -> list[str]:
        return list(texts)


class HFEntityMasker:
    """Masks people (and optionally places and organizations) with a multilingual NER model.

    Only worthwhile on text that keeps its capitalization. Long texts are cut into line-aligned
    chunks, long lines at whitespace (never mid-word); the pipeline's `stride` makes sure no
    tail is silently truncated at the model's token limit. `protected` terms (for example
    technologies the model mistakes for people) are never masked. Any failure raises.
    """

    PLACEHOLDERS = {"PER": "[NAME]", "LOC": "[LOC]", "ORG": "[ORG]"}

    def __init__(
        self,
        entity_types: Sequence[str] = ("PER", "LOC"),
        model: str = "Davlan/bert-base-multilingual-cased-ner-hrl",
        min_score: float = 0.35,
        batch_size: int = 16,
        chunk_chars: int = 800,
        protected: Iterable[str] = (),
        pipe: Any = None,  # injectable for tests
    ) -> None:
        self.entity_types = set(entity_types)
        self.model = model
        self.min_score = min_score
        self.batch_size = batch_size
        self.chunk_chars = chunk_chars
        self.protected = {fold(p) for p in protected}
        self.counts: Counter = Counter()
        self._pipe = pipe

    def _pipeline(self) -> Any:
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
            self._pipe = pipeline(
                "token-classification",
                model=self.model,
                aggregation_strategy="simple",
                device=device,
            )
        return self._pipe

    def _chunks(self, text: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for line in text.splitlines(keepends=True):
            while len(line) > self.chunk_chars:  # a single very long line: cut at whitespace
                cut = line.rfind(" ", 0, self.chunk_chars)
                cut = cut + 1 if cut > 0 else self.chunk_chars
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(line[:cut])
                line = line[cut:]
            if len(current) + len(line) > self.chunk_chars and current:
                chunks.append(current)
                current = ""
            current += line
        if current:
            chunks.append(current)
        return chunks

    def _unprotected_runs(self, chunk: str, start: int, end: int) -> list[tuple[int, int]]:
        """Split an entity span at protected words ("Analista Maria" masks only "Maria")."""
        runs: list[tuple[int, int]] = []
        first = last = None
        for m in _WORD.finditer(chunk, start, end):
            if fold(m.group()) in self.protected:
                if first is not None:
                    runs.append((first, last))  # type: ignore[arg-type]
                first = last = None
                continue
            first = m.start() if first is None else first
            last = m.end()
        if first is not None:
            runs.append((first, last))  # type: ignore[arg-type]
        return [r for r in runs if max(len(w) for w in _WORD.findall(chunk[r[0] : r[1]])) >= 3]

    def _mask_chunk(self, chunk: str, entities: list[dict]) -> str:
        spans = sorted(
            (run[0], run[1], e["entity_group"])
            for e in entities
            if e["entity_group"] in self.entity_types and e["score"] >= self.min_score
            for run in self._unprotected_runs(chunk, e["start"], e["end"])
        )
        for start, end, group in reversed(spans):
            chunk = chunk[:start] + self.PLACEHOLDERS[group] + chunk[end:]
            self.counts[group] += 1
        return chunk

    def mask_many(self, texts: Sequence[str]) -> list[str]:
        per_text = [self._chunks(t) for t in texts]
        flat = [c for chunks in per_text for c in chunks]
        if not flat:
            return list(texts)
        results = self._pipeline()(flat, batch_size=self.batch_size, stride=32)
        if len(results) != len(flat):
            raise RuntimeError("NER returned a different number of results than chunks")
        masked_flat = [self._mask_chunk(c, r) for c, r in zip(flat, results, strict=True)]
        out, i = [], 0
        for chunks in per_text:
            out.append("".join(masked_flat[i : i + len(chunks)]))
            i += len(chunks)
        return out


# --- orchestration over a column of texts -------------------------------------------------


def mask_texts(
    texts: Sequence[str],
    own_names: Sequence[str | None] | None = None,
    ner: EntityMasker | None = None,
    stats: Counter | None = None,
    dictionary: NameDictionary | None = None,
) -> list[str]:
    """Apply all layers to a batch of texts. Empty or missing texts stay empty."""
    names = own_names if own_names is not None else [None] * len(texts)
    stage = []
    for text, name in zip(texts, names, strict=True):
        masked = mask_own_name(mask_patterns(text or "", stats), name, stats)
        stage.append(mask_name_chains(masked, dictionary, stats) if dictionary else masked)
    unique = list(dict.fromkeys(stage))  # the NER model is the slow part: run each text once
    by_text = dict(zip(unique, (ner or NoEntityMasker()).mask_many(unique), strict=True))
    return [by_text[text] for text in stage]


# --- identifiers and generalization -------------------------------------------------------


def surrogate_id(kind: str, value: str | None, salt: bytes) -> str:
    """Stable pseudonymous id: HMAC-SHA256 of the value under a secret salt.

    Destroy the salt to make the mapping irreversible. Empty values stay empty (a value of
    "0" is a real value and is hashed like any other).
    """
    if value is None or value == "":
        return ""
    return hmac.new(salt, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:16]


_DATE_FORMATS = ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y")


def parse_date(value: str | None) -> datetime | None:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime((value or "").strip(), fmt)
        except ValueError:
            continue
    return None


def to_month(value: str | None) -> str | None:
    """Generalize a date to its month, 'YYYY-MM'."""
    parsed = parse_date(value)
    return parsed.strftime("%Y-%m") if parsed else None


def age_band(birth_date: str | None, snapshot: date) -> str | None:
    """Generalize a birth date to a 10-year age band, as attained on the snapshot date."""
    parsed = parse_date(birth_date)
    if parsed is None:
        return None
    age = (
        snapshot.year - parsed.year - ((snapshot.month, snapshot.day) < (parsed.month, parsed.day))
    )
    if not 14 <= age <= 80:
        return None  # implausible: treat as missing rather than keep a suspicious value
    if age < 25:
        return "<25"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    return "55+"
