# Modeling on the masked real data

## The data and the constraint

Three exports from an applicant tracking system (vacancies, candidates, and the candidacies that
link them), Portuguese-language, dated 2018-12 to 2025-03: 14,081 vacancies, 42,482 applicants,
53,759 candidacies. What shapes everything else:

* **77.7% of candidacy outcomes are still pending** (16.8% rejected or withdrawn, 5.6% hired). A
  pending candidacy is not a negative, so the labels are censored.
* **Structured applicant fields are 81-100% empty** (education, English, area, seniority). The
  CV text, present for 68% of applicants, carries almost all the signal. All CVs are lowercase.
* Only about 980 vacancies can be ranked at all (at least two candidates and one hire), and 490
  when only resolved outcomes are counted.
* The provider randomized the structured names, phones and emails, but real names, phones and
  addresses are still inside the CV text ([masking.md](masking.md) covers how they are handled).

The data cannot be published, so the architecture is a private store plus public code, aggregate
results and write-ups instead of "anyone can clone and run".

The loader (`src/masked_data.py`) reads only `data/masked/` and turns candidacies into labeled
(vacancy, candidate) pairs.

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

1. **Text match** (per pair): TF-IDF cosine on words and on character n-grams, how many of a
   vacancy's distinctive words appear in the CV and in the skills field, and the similarity of
   the applicant's professional title to the vacancy title. An optional multilingual-e5 cosine
   (the CV truncated to 512 tokens) via an injected encoder.
2. **Structured match**: education, English and Spanish gaps (ordinal), area match (exact and
   broad group), SAP vacancy against an SAP-area applicant, and whether the vacancy's seniority
   word appears in the CV. Applicant fields are 81-100% empty, so a "was filled in" indicator
   goes with each.
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
vacancies, only a large effect could be detected.

## Rolling-fold ranker (`src/ranker.py`)

```bash
make rank     # first cutoff 2021-06, six-month test windows
```

Each fold trains on every vacancy up to a cutoff, with labels as they stood then, and tests on
the next six months of vacancies with their final labels; every vacancy is tested once by a
model that only saw the past. That gives 244 test vacancies (resolved-only, 7 folds) and 474
(all-prospects, 8 folds) instead of the 57 and 125 of a single split. Models are pointwise
(a hire probability per candidate, ranked inside the vacancy): a regularized logistic and a
small gradient-boosted tree model. Pooled hit@1 gain over a random order, 95% bootstrap
interval over vacancies:

| Method | resolved-only | all-prospects |
|---|---|---|
| TF-IDF zero-shot | +2.2 [-4.0, +7.8] | +3.6 [-0.2, +7.2] |
| logistic | -4.2 [-9.6, +1.2] | +2.0 [-1.5, +5.4] |
| boosted trees | -3.8 [-9.5, +1.7] | +2.2 [-1.4, +5.9] |
| logistic, within-vacancy centered | -0.6 [-6.2, +4.9] | +4.1 [+0.4, +8.0] |
| boosted trees, within-vacancy centered | +1.9 [-3.7, +7.8] | **+6.4 [+2.7, +10.1]** |
| `lag_months` alone (process artifact) | +10.4 [+7.3, +13.9] | +4.5 [+2.6, +6.4] |

What the numbers do and do not say:

* **The raw pointwise models do no better than TF-IDF** (resolved-only: worse than random).
  Their largest weight is the number of candidates in the vacancy, a vacancy-level effect that
  cannot help rank candidates inside it. *Centering* every feature on its vacancy mean (a
  change made after seeing these first results, so it is one of about nine variants and should
  be read with that in mind) removes it.
* **The best clean model is the centered boosted one on all-prospects (+6.4)**, positive in 7 of
  its 8 folds, but its paired difference to plain TF-IDF is +2.8 [-1.5, +7.3]: not
  distinguishable from the zero-shot rule. On resolved-only, where the labels are cleaner, no
  model beats random and the per-fold signs flip.
* **The all-prospects label counts pending candidates as not hired**, so some "negatives" are
  hires that have not happened yet; that adds noise, and it may also favor features that track
  how far a candidate got in the funnel.
* **The strongest single signal is still the artifact**: `lag_months` alone is +10.4 on
  resolved-only, more than any model. It is excluded from the clean models; adding it back
  ("+ lag" rows in the report) does not help them either, since the pointwise model dilutes it.

Honest summary: the data supports a small text-matching signal (about +4 points of hit@1) that a
plain TF-IDF rule already captures; engineered structured features and a supervised model add,
at best, a few points that the available test vacancies cannot confirm.

## Two leaks found in review

Both let the future reach the training set, and fixing them removed the apparent signals of an
earlier draft (for example an education gap at +6 points of top-1 gain):

1. Selecting rankable vacancies *before* the split let a hire recorded after the cutoff decide
   what the training set contained. The split now comes first and the labels are rebuilt as of
   the cutoff.
2. The within-vacancy context was computed over a pool that depended on which candidates had
   been resolved. It is now computed over the whole pool, whatever the outcome of its members.

The guards are mutation-checked: each protection was deleted in turn and the tests failed.

## What I would do next

* A pairwise or lambda-rank objective, and the embedding feature (`make rank ARGS=--embed`).
* More outcomes: much of the censoring is recent, so the same pipeline would be re-run as
  outcomes resolve.
* Treat any result as a decision-support signal at most, never an automatic screen (see the
  README's Responsible use section).
