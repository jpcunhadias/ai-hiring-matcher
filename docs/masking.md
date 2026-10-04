# Masking the private data

The real challenge data (vacancies, prospects, applicants) comes from a hiring company and
contains personal data. It is **not published and cannot be**; this repository only ships
the code that handles it. `src/ingest.py` is the single place that reads the raw archives and
turns them into masked tables under `data/masked/` (gitignored). Everything downstream
(training, serving, the dashboard) reads only those masked tables.

```bash
make ingest                       # full run with the NER model (slow, run once)
uv run python -m src.ingest --ner none --limit 300 --review 20   # quick look
```

The raw zips are read straight into memory (nothing is extracted to disk), and the run is
**fail-closed**: any error aborts before a single file is written.

## What happens to each field

The output tables are built from an explicit **allowlist**. A field that is not named in
`src/ingest.py` is dropped, so forgetting one can never leak it.

| Source | Treatment |
|---|---|
| Applicant name, emails, phones, address, CPF, birth date, LinkedIn/Facebook/Skype, marital status, referral source, who inserted the record, salary expectation, institution and course names | **dropped** |
| Candidate id | **replaced** by a salted HMAC surrogate |
| Vacancy id, client | **replaced** by salted HMAC surrogates (the client name is dropped) |
| Sex, disability flag, birth date | moved to a **separate table** (`sensitive`), birth date generalized to a 10-year age band; for audits only, never features |
| Candidacy and requisition dates | generalized to the **month** |
| Recruiter, analyst, requester, client-contact names and phones; recruiter comments; vacancy names and objectives; neighborhood and city; commercial values | **dropped** |
| Education/language levels, area, seniority, contract type, priority, state, SAP flag, workflow status | **kept** as categories |
| CV text, skills, titles, certifications, vacancy activities and competencies | kept as text, **scrubbed** (below) |

## How free text is scrubbed

1. **Labeled lines** (`Estado civil:`, `Bairro:`, `Data de nascimento:`, `Nome:` ...): the value
   after the label is redacted, whatever it looks like.
2. **Patterns:** emails, URLs, CPF/CNPJ/RG, postcodes, full dates (numeric and written out),
   phone numbers, long digit runs, "N anos de idade". Year ranges such as `2015-2018` are left alone.
3. **The applicant's own name**, taken from the structured record, case- and accent-insensitive.
4. **Named-entity recognition** (multilingual BERT) for people and places. Organizations are
   *not* masked by default: the model also tags technologies such as SAP or Oracle as
   organizations, which would erase the skills this project matches on.

## What the tests guarantee

- A **canary test** plants a recognizable string in every sensitive field and asserts that none
  of them appears anywhere in any output table or in the report. It was mutation-checked:
  deliberately skipping the CV masking, or leaking a dropped field, makes it fail.
- The output schemas are exactly the allowlisted columns.
- Surrogate ids are stable, salted, namespaced, and not the original ids.
- A failing NER step writes nothing.

## What it does not guarantee

Masking lowers risk; it does not prove anonymity. Known residual risks:

- **Employer names** stay in the CV text (see above). Together with schools, dates and the
  state, a career history can be close to unique.
- **Names the NER misses**, and personal details in free prose that no label or pattern marks.
- **Quasi-identifier combinations.** `masking_report.json` reports how many applicants are
  unique on (education, seniority, area, age band, sex, disability); treat that as a risk
  indicator, not a guarantee.
- **The ids are pseudonymous, not anonymous.** Deleting `data/masked/.salt` makes the mapping
  back to the original ids unrecoverable; keeping it makes re-runs reproducible.
- Under Brazil's data protection law, pseudonymized data is still personal data. Publishing a
  masked version would also need the data owner's permission, which masking does not replace.

## Verifying a run

`data/masked/masking_report.json` holds aggregates only: row counts, replacement counts, residual
pattern hits per text field (all must be 0), surviving own-name hits (must be 0), the
share of empty structured fields, and the uniqueness figures. For a manual check,
`--review N` writes N masked CVs to `data/masked/review_sample.txt` so you can read them
yourself and look for anything the automation missed.
