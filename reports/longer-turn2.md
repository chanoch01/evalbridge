# Judge report card: `longer`

- Data: lmsys/mt_bench_human_judgments@f7d2896 (split `human`, turn 2)
- Items judged: 904 of 904 (80 questions); human votes scored: 1666
- Verdict used: merged across both orders
- 95% CIs: bootstrap over questions, 1000 resamples, seed 0

## Agreement with expert humans

| Measure | Value (95% CI) |
| --- | --- |
| Agreement, ties count | 53.5% (49.4% to 57.3%) |
| Agreement, both sides picked a winner | 68.5% (63.9% to 72.9%) (n=1272) |
| Cohen's kappa (A / B / tie) | 0.245 (0.188 to 0.301) |
| Agreement, single order as voters saw it | 53.5% (49.4% to 57.3%) |

## Position bias

| Measure | Value | Unbiased judge |
| --- | --- | --- |
| Same verdict after swapping order | 100.0% | 100% |
| First-shown answer wins (decided verdicts) | 50.0% | 50% |

## Length bias

Decided verdicts on pairs of unequal length (word count of the judged answer).

| Who | Prefers the longer answer |
| --- | --- |
| This judge | 100.0% |
| Expert humans | 68.5% |

## Reference points (same items)

| Reference | Agreement | Agreement, single order | Agreement, no ties | Kappa | Swap-consistent | First wins | Prefers longer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human vs human (1169 vote pairs) | 67.0% | | 83.4% | | | | 68.5% |
| `longer` baseline | 53.5% | 53.5% | 68.5% | 0.245 | 100.0% | 50.0% | 100.0% |
| `first` baseline | 22.5% | 38.2% | n/a | 0.000 | 0.0% | 100.0% | n/a |

Ties: a tie agrees only with a tie. Kappa uses three classes. Order-merged verdicts become a tie when the two orders disagree.
