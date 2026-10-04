# Masking the private data

The real challenge data (vacancies, prospects, applicants) comes from a hiring company and
contains personal data. It is **not published and cannot be**; this repository only ships
the code that handles it. `src/ingest.py` is the single place that reads the raw archives and
turns them into masked tables under `data/masked/` (gitignored, directory `0700`, files `0600`).

> **Status:** ingest is a separate stage. Training, serving and the dashboard still read the
> public Kaggle CSV; wiring them to the masked tables is the next step, not done yet.

```bash
make ingest                       # full run from the raw archives
make repair                       # re-apply the masking layers to the existing masked tables
uv run python -m src.ingest --ner none --limit 300 --review 20   # quick look
```

The raw zips are read straight into memory (nothing is extracted to disk). The run is
**fail-closed and atomic**: it builds a complete private staging generation and only then
swaps it in, so an error never leaves new files next to old ones, and the previous generation
is kept as `.previous` until the swap succeeds.

## What happens to each field

The output tables are built from an explicit **allowlist**. A field that is not named in
`src/ingest.py` is dropped, so forgetting one can never leak it.

| Source | Treatment |
|---|---|
| Applicant name, emails, phones, address, CPF, birth date, LinkedIn/Facebook/Skype, marital status, referral source, who inserted the record, salary expectation, institution and course names | **dropped** |
| Candidate id | **replaced** by a salted HMAC surrogate (a numeric id of `0` is kept distinct from "missing") |
| Vacancy id, client | **replaced** by salted HMAC surrogates (the client name is dropped) |
| Sex, disability flag, birth date | moved to a **separate table** (`sensitive`), birth date generalized to a 10-year age band computed at the snapshot date; for audits only, never features |
| Candidacy and requisition dates | generalized to the **month** |
| Recruiter, analyst, requester, client-contact names and phones; recruiter comments; vacancy names and objectives; neighborhood and city; commercial values | **dropped** |
| Education/language levels, seniority, contract type, priority, state, SAP flag, workflow status | **kept** as categories |
| CV text, skills, area, certifications, titles, vacancy activities and competencies | kept as text, **scrubbed** (below) |

## How free text is scrubbed

Layers run in this order on every free-text field:

1. **Labeled lines** (`Estado civil:`, `Bairro:`, `Nome`, `Endereço =` ...): the value after the
   label is redacted, whether it follows a separator, sits on the next line, or is a bare
   number (`idade 32`, `rg 12 345 678 9`).
2. **Patterns:** emails, URLs, CPF/CNPJ/RG, postcodes, dates (numeric, ISO, written out, also
   glued to a letter), phone numbers (including spaced, en-dash and "(" -less ones), long digit
   runs, marital status and children ("sem filhos", "2 filhos"), street addresses
   (`[ADDRESS]`) and neighborhood names (`[LOCATION]`; the city and state are kept). Ages are
   masked in a personal-data context only ("nascida em [DATE], 34 anos."); "mais de 10 anos
   de experiência" stays. Year ranges such as `2015-2018` are left alone.
3. **The applicant's own name**, from the structured record: accent-insensitive,
   whole-phrase, also for short names.
4. **Name chains from a dictionary.** A run of 2–4 tokens on one line that starts with a
   common Brazilian first name and continues with name tokens is masked. The dictionary is
   the 3,000 most frequent first names of the IBGE census plus the name tokens that occur in
   the source system's own name fields. In the first 400 characters of a text (where a
   person's own name sits), up to two *unknown* words after a first name also continue the
   chain, as unknown surnames: a word is unknown when it is not an ordinary word of the
   vacancy vocabulary. Deeper in a text this is off, because there an unknown word after a
   first name is more often a school or employer. `santo`/`santa`/`são` never start a chain
   (`Santo André` is a place). Commas and line breaks end a chain.
5. **Named-entity recognition** (multilingual BERT), only on short fields that keep their
   capitalization (titles, professional title). Spans are split at technology and job words
   that appear in many documents, so `Analista Maria` masks only `Maria`. Organizations are
   *not* masked: the model also tags technologies such as SAP or Oracle as organizations,
   which would erase the skills this project matches on.

### Why NER is not used on the CV text

The CVs in this data set are 100% lowercase, and a cased NER model cannot recognize names
without capitals. On synthetic names injected into real masked CV text it recovered an
average of **13.7%** of name tokens; the dictionary chains recovered **about 90%**
(full names: 94.8%) with a few dozen false positives per 600 text chunks, against hundreds
for the unrestricted 64k-name list. NER is also the slow part (about 70 chunks/s on a laptop
GPU for the 172k CV chunks), so dropping it for CVs turns a 45-minute run into minutes.
Evaluated and rejected: Presidio (a framework with no Portuguese configuration of its own),
piiranha (no Portuguese), `gliner_multi_pii` (highest precision, but recall around 0.3 here).
These numbers come from synthetic names, not from labelled real ones, and they overstated the
dictionary: real surnames are often not in any list, so a real CV with "marcela lucindo" kept
its name until the unknown-surname rule above was added (found by an outside review of the
review sample). Treat the recall as unknown, somewhere below 90%, for names outside the header.

## `--repair`

Masking layers are idempotent: masked text passes through them unchanged. `--repair` loads the
existing masked tables, re-applies the current layers, re-reads the raw archive only for the
structured names, the dictionary and the birth dates, and publishes a new generation. Use it
when a layer is added or fixed; it takes a few minutes instead of a full re-ingest.

## What the tests guarantee

- A **canary test** plants raw identifiers (phones, CPF, birth date, names, emails) in every
  sensitive field and asserts that none of them appears anywhere in any output surface:
  tables, report, review sample. It was mutation-checked.
- The output schemas are exactly the allowlisted columns.
- Surrogate ids are stable, salted, namespaced, and not the original ids.
- A failing NER step writes nothing; a failure mid-publish keeps the previous generation.
- Output permissions are `0700`/`0600` regardless of the caller's umask; a missing, empty,
  short or world-readable salt is rejected or tightened.
- `--repair` is idempotent.

## What it does not guarantee

Masking lowers risk; it does not prove anonymity. Known residual risks:

- **Employer names** stay in the CV text (see above). Together with schools, dates and the
  state, a career history can be close to unique.
- **Names the dictionary does not know**, a first name written on its own, and personal details
  in free prose that no label or pattern marks.
- **Quasi-identifier combinations.** `masking_report.json` reports how many applicants are
  unique on (education, seniority, area, age band, sex, disability); treat that as a risk
  indicator, not a guarantee.
- **The ids are pseudonymous, not anonymous.** Deleting `data/masked/.salt` makes the mapping
  back to the original ids unrecoverable; keeping it makes re-runs reproducible.
- Under Brazil's data protection law, pseudonymized data is still personal data. Publishing a
  masked version would also need the data owner's permission, which masking does not replace.

## Verifying a run

`data/masked/masking_report.json` holds aggregates only. It has two kinds of residual figures,
and they mean different things:

- `self_consistency_residual_hits` re-runs the masker's *own* patterns. It must be 0 and only
  proves the masker agrees with itself.
- `independent_residual_estimate` uses separately written, looser checks and estimates what may
  still have slipped through. It is not zero: part of it is false positives (employer
  domains such as `x.com.br`, `analista @ empresa`, 8-digit numbers in vacancy titles, year
  ranges split across lines that look like phone numbers, and job-tenure lines such as
  `12 anos` on their own line that look like ages). On the real CVs the repair took the
  independent hits for ISO dates, labeled values, civil status, label-then-text lines and
  neighborhood-like phrases to 0 or near 0, and cut the date and phone-like hits roughly in half.

For a manual check, `--review N` writes N masked CVs to `data/masked/review_sample.txt` so you
can read them yourself and look for anything the automation missed.

## Reference data

`data/reference/nomes-censos-ibge.csv` (gitignored) is the public IBGE first-name list from
the 2010 census. It carries no license statement, so it is fetched locally and not
redistributed; set `FIRST_NAMES_CSV` to use another path.
