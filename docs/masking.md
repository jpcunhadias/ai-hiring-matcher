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

The raw zips are read straight into memory (nothing is extracted to disk). A run is
**fail-closed on errors and on detected residue**, and publishes in two steps:

* Everything is built in a private staging directory first. An error, or a **privacy gate**
  failure (below), leaves the previous generation untouched and writes no salt file.
* Publishing is two renames, not one atomic operation: the live directory becomes `.previous`,
  then the staging directory takes its place. Between the two there is no live directory. A
  crash there is recovered automatically on the next run or load (`.previous` is put back, so
  the salt it holds is not lost) and stale staging directories are removed. One writer at a time
  is assumed. Every generation carries a `GENERATION` marker; the loader reads it before and
  after reading the four tables and retries if it changed, so it never mixes two generations.

### The privacy gate

Before anything is published, the finished tables are checked and publication is **blocked**
(with counts only in the message) if any of these is non-zero:

* the masker's own patterns still match its output (self-consistency);
* the applicant's own name survives in a CV;
* the independent checks find an ISO date, a labeled personal-data line, marital status or
  children, or an email-like token (`name@domain`, with or without a top-level domain).

The softer estimates (domains, spaced numbers, "NN anos" tenure lines) have known false
positives and are reported, not enforced. The gate enforces invariants a finished table must
satisfy; it cannot make heuristic masking exhaustive. A phone number in a format no pattern
knows would pass it. The name list is also required: without the first-name file the run stops
instead of silently switching the name layer off (`--no-name-dictionary` waives it, and the
report then says so).

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
| Education/language levels, seniority, contract type, priority, state, SAP flag, workflow status, modality, sex, disability flag | **kept** as categories, and **validated**: a value that carries anything a pattern would mask (an email, phone, date, labeled line), an `@`, or is longer than 120 characters is replaced by `[REDACTED]`; the report counts replacements per column |
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
   whole-phrase, also for short names. The record can hold the name in two fields; each distinct
   spelling is masked on its own (never glued into one phrase, which would match nothing).
4. **Name chains from a dictionary.** A run of 2–4 tokens on one line that starts with a
   common Brazilian first name and continues with name tokens is masked. The dictionary is
   the 3,000 most frequent first names of the IBGE census plus the name tokens that occur in
   the source system's own name fields. In the first 400 characters of a text (where a
   person's own name sits), up to two *unknown* words after a first name also continue the
   chain, as unknown surnames: a word is unknown when it is not an ordinary word of the
   vacancy vocabulary. An unknown *first* name is caught when two known surnames follow it, and
   in the first 150 characters one line break between name words is tolerated (PDF text
   often has one word per line). Deeper in a text this is off, because there an unknown word after a
   first name is more often a school or employer. `santo`/`santa`/`são` never start a chain
   (`Santo André` is a place). Commas and line breaks end a chain.
5. **Named-entity recognition** (multilingual BERT), only on fields that keep their
   capitalization: vacancy titles, activities and competencies, and the applicant's professional
   title. Spans are split at protected words, so `Analista Maria` masks only `Maria`. Protected
   words are seniority and role words plus generic tokens that are frequent in the field *and
   are not personal-name tokens*: frequency alone never protects a word, because a frequent first
   name would then escape the model. Organizations are
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

Masking layers are idempotent in practice: masked text passes through them unchanged, apart from
rare ordering effects (a later layer can expose a match for an earlier one), so after changing a
layer run `--repair` until the replacements this pass drop to nearly zero. `--repair` loads the
existing masked tables, re-applies the current layers, re-reads the raw archive only for the
structured names, the dictionary and the birth dates, and publishes a new generation. Use it
when a layer is added or fixed; it takes a few minutes instead of a full re-ingest.

## What the tests guarantee

- A **canary test** plants raw identifiers (phones, CPF, birth date, names, emails) in every
  sensitive field and asserts that none of them appears anywhere in any output surface:
  tables, report, review sample. It was mutation-checked.
- The output schemas are exactly the allowlisted columns.
- Surrogate ids are stable, salted, namespaced, and not the original ids.
- A failing NER step writes nothing; a failure mid-publish keeps the previous generation; a
  crash between the two renames is recovered.
- The privacy gate blocks publication on any hard residual, and a missing name list stops the run.
- Retained categories that carry personal data are replaced; a short name present in both name
  fields is masked and not reported as surviving.
- Output permissions are `0700`/`0600` regardless of the caller's umask; a missing, empty,
  short or world-readable salt is rejected or tightened. A salt given in the environment must
  agree with the stored one.
- `--repair` refuses to run without the original salt, and refuses a salt that does not
  reproduce the ids already published (it never invents one); it is idempotent.
- Errors raised while masking carry no text from the data: the message names the stage and the
  exception type only, the original is not chained, and the command line withholds details of
  unexpected errors (`INGEST_DEBUG=1` shows them for debugging on synthetic data).
- Every protection above was mutation-checked: deleting it makes a test fail.

## What it does not guarantee

Masking lowers risk; it does not prove anonymity, and no heuristic masker can promise output free
of every identifier. Withholding the free text altogether would, but it is the signal. Known
residual risks:

- **Formats nobody wrote a pattern for**: the gate enforces specific invariants, not "no
  personal data".
- **Third-party libraries' error text**: confidentiality of error messages is enforced for this
  code; whether the real NER dependency ever echoes its input in its own messages has not been
  established, which is why exceptions are not chained or printed.
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
