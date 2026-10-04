"""Masking primitives for the private ingest of the real ATS data.

Everything here is pure and unit-tested. The principle throughout is fail-closed: if
something can't be masked, it raises, and the ingest writes nothing.

Layers applied to free text, in order:
1. patterns (email, URL, ID numbers, postcode, full dates, phone, long digit runs)
2. the applicant's own name, taken from the structured record
3. named-entity recognition for people and places (organizations are opt-in, because the
   model also tags technologies such as "SAP" or "Oracle" as organizations)
"""

import hashlib
import hmac
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol

# --- layer 1: patterns --------------------------------------------------------------------

_YEAR_RANGE = re.compile(r"(?:19|20)\d{2}[\s.-](?:19|20)\d{2}")

# Phone numbers are matched by several explicit shapes instead of one permissive regex,
# because a permissive one also eats year ranges such as "2015-2018".
_PHONE = re.compile(
    r"(?:\+?55[\s.-]?)?\(\s?\d{2}\s?\)[\s.-]?9?[\s.-]?\d{4}[\s.-]?\d{4}"  # (11) 98765-4321
    r"|\+55[\s.-]?\d{2}[\s.-]?9?[\s.-]?\d{4}[\s.-]?\d{4}"  # +55 11 98765 4321
    r"|(?<![\d-])\d{2}[\s.-]9?\d{4}[\s.-]\d{4}(?![\d-])"  # 11 98765-4321
    r"|(?<![\d-])9?\d{4}-\d{4}(?![\d-])"  # 98765-4321, 3333-4444
)

# CVs often carry labeled personal-data lines ("Estado civil: casado", "Bairro: ..."). The
# value after such a label is redacted, whatever it looks like: patterns can't recognise a
# neighborhood or "12 de março de 1990". Address-like labels take the rest of the line;
# short attributes stop at the next separator so "Idade: 32, Solteiro" keeps what follows.
_LINE_LABELS = (
    r"endere[cç]o|bairro|cidade|cep|telefone|celular|tel|fone|e-?mail|contato"
    r"|nome(?: completo)?|filia[cç][aã]o|pai|m[aã]e|skype"
)
_SHORT_LABELS = (
    r"data de nascimento|nascimento|idade|estado civil|naturalidade|nacionalidade|sexo"
    r"|g[eê]nero|ra[cç]a|cor|rg|cpf|cnh|filhos|dependentes"
)
_SEP = r"[ \t]*[:\-–][ \t]*"
_LABELED = [
    # label alone on a line, value on the next line
    re.compile(
        rf"(?<!\w)((?:{_LINE_LABELS}|{_SHORT_LABELS}){_SEP}\n[ \t]*)([^\n]+)", re.IGNORECASE
    ),
    re.compile(rf"(?<!\w)((?:{_LINE_LABELS}){_SEP})([^\n]*)", re.IGNORECASE),
    re.compile(rf"(?<!\w)((?:{_SHORT_LABELS}){_SEP})([^,;|\n]*)", re.IGNORECASE),
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
        "ID",  # CPF, CNPJ, RG
        re.compile(
            r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"
            r"|\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b"
            r"|\b\d{1,2}\.\d{3}\.\d{3}-[\dxX]\b"
        ),
    ),
    ("POSTCODE", re.compile(r"\b\d{5}-\d{3}\b")),
    ("DATE", re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)?\d{2}\b")),  # full dates only
    ("DATE", re.compile(rf"\b\d{{1,2}} de (?:{_PT_MONTHS}) de (?:19|20)\d{{2}}\b", re.IGNORECASE)),
    ("AGE", re.compile(r"\b\d{2} anos de idade\b", re.IGNORECASE)),
    ("PHONE", _PHONE),
    ("NUMBER", re.compile(r"(?<!\d)\d{9,}(?!\d)")),
]


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


def _labeled_residual(text: str) -> bool:
    return any(
        m.group(2).strip() not in ("", _REDACTED)
        for pattern in _LABELED
        for m in pattern.finditer(text)
    )


def mask_patterns(text: str, stats: Counter | None = None) -> str:
    """Redact labeled personal-data lines, then emails, URLs, IDs, postcodes, dates, phones."""
    text = _redact_labeled(text, stats)
    for label, pattern in _PATTERNS:

        def repl(match: re.Match[str], label: str = label) -> str:
            if label == "PHONE" and _YEAR_RANGE.fullmatch(match.group().strip(" ()")):
                return match.group()  # a year range, not a phone number
            if stats is not None:
                stats[label] += 1
            return f"[{label}]"

        text = pattern.sub(repl, text)
    return text


def residual_pattern_counts(texts: Sequence[str]) -> dict[str, int]:
    """How many of the (already masked) texts still contain each pattern. Should be all 0."""
    counts = {label: 0 for label, _ in _PATTERNS} | {"LABELED": 0}
    for text in texts:
        counts["LABELED"] += _labeled_residual(text)
        for label, pattern in _PATTERNS:
            for match in pattern.finditer(text):
                if label == "PHONE" and _YEAR_RANGE.fullmatch(match.group().strip(" ()")):
                    continue
                counts[label] += 1
                break
    return counts


# --- layer 2: the applicant's own name ----------------------------------------------------

_NAME_PARTICLES = {"da", "de", "do", "das", "dos", "e", "di", "du", "del", "van", "von"}


def _strip_accents(value: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", value) if unicodedata.category(c) != "Mn"
    )


def name_tokens(name: str | None) -> list[str]:
    """Distinctive tokens of a person's name (no particles, nothing under 3 letters)."""
    if not name:
        return []
    words = re.findall(r"[^\W\d_]+", name, flags=re.UNICODE)
    return [w for w in words if len(w) >= 3 and w.lower() not in _NAME_PARTICLES]


def mask_own_name(text: str, name: str | None, stats: Counter | None = None) -> str:
    for token in name_tokens(name):
        variants = {re.escape(token), re.escape(_strip_accents(token))}
        pattern = re.compile(rf"(?<!\w)(?:{'|'.join(sorted(variants))})(?!\w)", re.IGNORECASE)
        text, n = pattern.subn("[NAME]", text)
        if stats is not None and n:
            stats["OWN_NAME"] += n
    return text


# --- layer 3: named-entity recognition -----------------------------------------------------


class EntityMasker(Protocol):
    counts: Counter

    def mask_many(self, texts: Sequence[str]) -> list[str]: ...


class NoEntityMasker:
    """Identity masker: used for short structured fields and for fast runs and tests."""

    def __init__(self) -> None:
        self.counts: Counter = Counter()

    def mask_many(self, texts: Sequence[str]) -> list[str]:
        return list(texts)


class HFEntityMasker:
    """Masks people (and optionally places and organizations) with a multilingual NER model.

    Long texts are cut into line-aligned chunks; the pipeline's `stride` makes sure no
    tail is silently truncated at the model's 512-token limit. Any failure raises.
    """

    PLACEHOLDERS = {"PER": "[NAME]", "LOC": "[LOC]", "ORG": "[ORG]"}

    def __init__(
        self,
        entity_types: Sequence[str] = ("PER", "LOC"),
        model: str = "Davlan/bert-base-multilingual-cased-ner-hrl",
        min_score: float = 0.35,
        batch_size: int = 16,
        chunk_chars: int = 800,
        pipe: Any = None,  # injectable for tests
    ) -> None:
        self.entity_types = set(entity_types)
        self.model = model
        self.min_score = min_score
        self.batch_size = batch_size
        self.chunk_chars = chunk_chars
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
            while len(line) > self.chunk_chars:  # a single very long line
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(line[: self.chunk_chars])
                line = line[self.chunk_chars :]
            if len(current) + len(line) > self.chunk_chars and current:
                chunks.append(current)
                current = ""
            current += line
        if current:
            chunks.append(current)
        return chunks

    def _mask_chunk(self, chunk: str, entities: list[dict]) -> str:
        spans = sorted(
            (e["start"], e["end"], e["entity_group"])
            for e in entities
            if e["entity_group"] in self.entity_types and e["score"] >= self.min_score
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
) -> list[str]:
    """Apply all layers to a batch of texts. Empty or missing texts stay empty."""
    names = own_names if own_names is not None else [None] * len(texts)
    stage = [
        mask_own_name(mask_patterns(t or "", stats), n, stats)
        for t, n in zip(texts, names, strict=True)
    ]
    return (ner or NoEntityMasker()).mask_many(stage)


# --- identifiers and generalization -------------------------------------------------------


def surrogate_id(kind: str, value: str | None, salt: bytes) -> str:
    """Stable pseudonymous id: HMAC-SHA256 of the value under a secret salt.

    Destroy the salt to make the mapping irreversible. Empty values stay empty.
    """
    if not value:
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


def age_band(birth_date: str | None, reference_year: int = 2025) -> str | None:
    """Generalize a birth date to a 10-year age band as of the data snapshot."""
    parsed = parse_date(birth_date)
    if parsed is None:
        return None
    age = reference_year - parsed.year
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
