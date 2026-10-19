# Judge report card: `gpt4-recorded`

- Data: lmsys/mt_bench_human_judgments@f7d2896 (split `human`, turn 2)
- Items judged: 904 of 904 (80 questions); human votes scored: 1666
- Verdict used: single order
- 95% CIs: bootstrap over questions, 1000 resamples, seed 0

## Agreement with expert humans

| Measure | Value (95% CI) |
| --- | --- |
| Agreement, ties count | 65.7% (62.0% to 69.3%) |
| Agreement, both sides picked a winner | 85.3% (81.9% to 88.5%) (n=1069) |
| Cohen's kappa (A / B / tie) | 0.474 (0.420 to 0.527) |

## Position bias

Not measured: this judge's verdicts do not depend on order (recorded verdicts were already merged across both orders).
Recorded labels in the source split: `model_a` 190, `model_b` 1610, `tie` 220, `tie (inconsistent)` 380.

## Length bias

Decided verdicts on pairs of unequal length (word count of the judged answer).

| Who | Prefers the longer answer |
| --- | --- |
| This judge | 73.7% |
| Expert humans | 68.5% |

## Reference points (same items)

| Reference | Agreement | Agreement, single order | Agreement, no ties | Kappa | Swap-consistent | First wins | Prefers longer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human vs human (1169 vote pairs) | 67.0% | | 83.4% | | | | 68.5% |
| `longer` baseline | 53.5% | 53.5% | 68.5% | 0.245 | 100.0% | 50.0% | 100.0% |
| `first` baseline | 22.5% | 38.2% | n/a | 0.000 | 0.0% | 100.0% | n/a |

Ties: a tie agrees only with a tie. Kappa uses three classes. Order-merged verdicts become a tie when the two orders disagree.
