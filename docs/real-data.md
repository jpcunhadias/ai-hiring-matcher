# Modeling on the masked real data

The loader (`src/masked_data.py`) reads only `data/masked/` (see [masking.md](masking.md)) and
turns candidacies into labeled (vacancy, candidate) pairs.

```bash
make data-summary     # aggregates per label setting and the temporal split
```

## Label

`y = 1` when the candidacy ended in a hire. 77.7% of outcomes are still pending, and a pending
candidacy is not a negative, so two settings are reported side by side:

| Setting | Candidacies counted | Trade-off |
|---|---|---|
| `resolved_only` | outcome known (hired or rejected/withdrew) | honest labels, small: 493 rankable vacancies |
| `all_prospects` | everything, pending as "not hired" | larger (978 vacancies) but noisier: some pending candidates may still be hired |

A vacancy is rankable when it has at least two candidates and at least one hire (and, resolved
only, at least one candidate who was passed over). Ranking is judged *within* a vacancy.

## Split

Train on earlier vacancies, test on later ones. The cutoff is the requested month below which
80% of the vacancies with at least two candidates fall (no outcome influences it), or an
explicit `--cutoff YYYY-MM`. The split comes first; the training labels are then rebuilt *as of
the cutoff*:

* a candidacy that started after the cutoff did not exist yet and is dropped;
* an outcome recorded after the cutoff, or with no recorded date, counts as still pending;
* only then are the rankable vacancies chosen, so no future outcome decides what the training
  set contains.

The test period keeps its final labels, because that is what is being predicted. Recent
vacancies have more pending outcomes and therefore fewer rankable ones, so the test sets are
small (57 resolved-only and 125 all-prospects vacancies at the 2023-01 cutoff): confidence
intervals are wide, and results should be read as "no clear signal" rather than as estimates.

The within-vacancy context (candidate count, z-score and rank of each score) is computed over
the vacancy's whole pool, whatever the outcome of its members, so it cannot depend on who was
resolved. It describes the completed pool: the evaluation is a retrospective ranking of that
pool, not a replay of the moment each candidate applied.

## What can and cannot be a feature

* Sex, disability and age band are in a separate table and never joined into the pairs. They
  are for fairness audits only (`audit_attributes`).
* `status` and `updated_month` describe the outcome itself (a later funnel stage is a hire
  proxy); they are used for the label and the split, never as features.
* The ids are salted hashes; they identify, they do not describe.

## Features (`src/features.py`)

1. **Text match** (per pair): TF-IDF cosine on words and on character n-grams, BM25, and
   multilingual-e5 cosine (CV chunked, best chunk against the vacancy); overlap between the
   applicant's skills and the vacancy competencies; similarity of professional title and
   vacancy title.
2. **Structured match**: seniority, education and English gaps (ordinal), area match, SAP
   vacancy against an SAP-area applicant, state. Applicant fields are 81-100% empty, so a
   "was filled in" indicator goes with each.
3. **Within-vacancy context**: each score as a z-score and rank among that vacancy's
   candidates, the number of candidates, CV length (a confound check: length alone showed no signal).
4. **History, computed only from the past**: an applicant's earlier candidacies and hires and a
   client's earlier hire rate. A row at month t only sees candidacies started before t, and an
   outcome only once it had been recorded before t.

`FeatureBuilder` is fit on the training period only (TF-IDF vocabularies, the list of generic
vacancy words, the global hire rate) and then transforms any frame. Optional multilingual-e5
cosine via an injected encoder (`make feature-report ARGS=--embed`).

### A signal that is not a signal: `lag_months`

The months between the vacancy request and the candidacy was the strongest single feature on
the test period (+10.8 points of hit@1 over random, resolved-only). It is funnel dynamics, not
candidate quality: recruiters keep adding candidates until someone is hired, so the hire tends
to be the latest one added (68% of the time in resolved vacancies against a 40% base rate). It
would not exist when ranking a fresh pool. It stays in the feature frame as a diagnostic and is
excluded by `model_features()` unless asked for as an ablation.

### Single-feature results

`make feature-report` evaluates each feature alone as a ranking rule on the later vacancies
(hit@1 gain over a random order, 95% bootstrap interval over vacancies, exact tie handling).

At the 2023-01 cutoff **no candidate-quality feature separates from random**: every interval
includes zero except `lag_months` (+10.9 points resolved-only), the process artifact above, and
a seniority-word-in-CV cue that points the wrong way (-7.9). TF-IDF is +1.8 and -2.7 points. An
earlier version of this pipeline showed a few apparent signals (for example an education gap at
+6 points); they did not survive fixing two leaks, in which the choice of training vacancies and
the within-vacancy context depended on outcomes recorded after the cutoff. Both fixes and the
tests that guard them are described in the commit history.

Consequences for the project: the zero-shot text similarity measured over all vacancies (about
+4 points) is weak and does not clearly persist on later vacancies; and with this few test
vacancies, only a large effect could be detected. The next step is a supervised ranker
evaluated over several rolling cutoffs, which yields more test vacancies.
