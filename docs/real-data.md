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

## Planned features

1. **Text match** (per pair): TF-IDF cosine on words and on character n-grams, BM25, and
   multilingual-e5 cosine (CV chunked, best chunk against the vacancy); overlap between the
   applicant's skills and the vacancy competencies; similarity of professional title and
   vacancy title.
2. **Structured match**: seniority, education and English gaps (ordinal), area match, SAP
   vacancy against an SAP-area applicant, state. Applicant fields are 81-100% empty, so a
   "was filled in" indicator goes with each.
3. **Within-vacancy context**: each score as a z-score and rank among that vacancy's
   candidates, the number of candidates, CV length (a confound check: length alone showed no signal).
4. **History, computed only from the past**: an applicant's earlier candidacies and hires, the
   months between the vacancy request and the application, a client's earlier hire rate.

Models, in order: the TF-IDF zero-shot baseline (about +4 points of hit@1 over random), a
logistic pairwise ranker, then gradient-boosted ranking if the features earn it.
