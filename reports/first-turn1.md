# Judge report card: `first`

- Data: lmsys/mt_bench_human_judgments@f7d2896 (split `human`, turn 1)
- Items judged: 910 of 910 (80 questions); human votes scored: 1689
- Verdict used: merged across both orders
- 95% CIs: bootstrap over questions, 1000 resamples, seed 0

## Agreement with expert humans

| Measure | Value (95% CI) |
| --- | --- |
| Agreement, ties count | 24.0% (20.4% to 27.9%) |
| Agreement, both sides picked a winner | n/a (n=0) |
| Cohen's kappa (A / B / tie) | 0.000 (0.000 to 0.000) |
| Agreement, single order as voters saw it | 38.8% (36.1% to 41.3%) |

## Position bias

| Measure | Value | Unbiased judge |
| --- | --- | --- |
| Same verdict after swapping order | 0.0% | 100% |
| First-shown answer wins (decided verdicts) | 100.0% | 50% |

## Length bias

Decided verdicts on pairs of unequal length (word count of the judged answer).

| Who | Prefers the longer answer |
| --- | --- |
| This judge | n/a |
| Expert humans | 68.9% |

## Reference points (same items)

| Reference | Agreement | Agreement, single order | Agreement, no ties | Kappa | Swap-consistent | First wins | Prefers longer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Human vs human (1198 vote pairs) | 64.1% | | 82.0% | | | | 68.9% |
| `longer` baseline | 53.5% | 53.5% | 68.9% | 0.255 | 100.0% | 50.0% | 100.0% |
| `first` baseline | 24.0% | 38.8% | n/a | 0.000 | 0.0% | 100.0% | n/a |

Ties: a tie agrees only with a tie. Kappa uses three classes. Order-merged verdicts become a tie when the two orders disagree.
