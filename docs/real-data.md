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

Train on earlier vacancies, test on later ones (cutoff on the requested month, 80/20 by
vacancies). A training row is dropped when its outcome was recorded after the cutoff: in
production that label would not exist yet. Test sets are small (about 90 and 180 vacancies), so
confidence intervals are wide; say so when reporting.

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

`make feature-report` evaluates each feature alone as a ranking rule on the later vacancies.
The test sets are small (92 and 177 vacancies), so most intervals include zero; read the
ranking of features, not the point estimates. Text similarity alone (TF-IDF) is about +3 and
+0 points here, below the +4 measured over all vacancies earlier, which suggests the
zero-shot signal is weak and drifts over time. The next step is a supervised ranker evaluated
over several rolling time folds to get more test vacancies.

Models, in order: the TF-IDF zero-shot baseline (about +4 points of hit@1 over random), a
logistic pairwise ranker, then gradient-boosted ranking if the features earn it.
