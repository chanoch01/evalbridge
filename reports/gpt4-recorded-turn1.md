# Judge report card: `gpt4-recorded`

- Data: lmsys/mt_bench_human_judgments@f7d2896 (split `human`, turn 1)
- Items judged: 910 of 910 (80 questions); human votes scored: 1689
- Verdict used: single order
- 95% CIs: bootstrap over questions, 1000 resamples, seed 0

## Agreement with expert humans

| Measure | Value (95% CI) |
| --- | --- |
| Agreement, ties count | 66.6% (62.8% to 70.3%) |
| Agreement, both sides picked a winner | 85.9% (82.7% to 89.0%) (n=1060) |
| Cohen's kappa (A / B / tie) | 0.491 (0.434 to 0.549) |

## Position bias

Not measured: this judge's verdicts do not depend on order (recorded verdicts were already merged across both orders).
Recorded labels in the source split: `model_a` 190, `model_b` 1610, `tie` 220, `tie (inconsistent)` 380.

## Length bias

Decided verdicts on pairs of unequal length (word count of the judged answer).

| Who | Prefers the longer answer |
| --- | --- |
| This judge | 74.6% |
| Expert humans | 68.9% |

## Reference points (same items)

| Reference | Agreement | Agreement, single order | Agreement, no ties | Kappa | Swap-consistent | First wins | Prefers longer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human vs human (1198 vote pairs) | 64.1% | | 82.0% | | | | 68.9% |
| `longer` baseline | 53.5% | 53.5% | 68.9% | 0.255 | 100.0% | 50.0% | 100.0% |
| `first` baseline | 24.0% | 38.8% | n/a | 0.000 | 0.0% | 100.0% | n/a |

Ties: a tie agrees only with a tie. Kappa uses three classes. Order-merged verdicts become a tie when the two orders disagree.
