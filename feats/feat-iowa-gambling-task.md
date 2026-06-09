# Four-Deck Card Task

The built-in `four-deck-card-task` is an LLM-oriented implementation of classic
Iowa Gambling Task contingencies.

## Model-Facing Protocol

The model is told that it must repeatedly choose one of four card options to
maximize its final balance. It is not shown:

- The name "Iowa Gambling Task".
- Canonical A-D deck labels.
- Internal contingency IDs.
- Advantageous classifications.
- Future gains or penalties.

Visible labels are deterministically counterbalanced per subject and their
hidden mapping is stored with the resolved schedule.

## Default Contingencies

The defaults preserve the classic structure:

- Two decks pay 100 per draw and are disadvantageous over repeated draws.
- Two decks pay 50 per draw and are advantageous over repeated draws.
- Within each pair, one deck has frequent smaller losses and one has rare large
  losses.

Penalty lists are explicit editable templates. In `template` mode they repeat
to cover the requested trial count and may be shuffled within each repeated
block. In `fixed` mode every deck must provide at least `trial_count` penalties.

The defaults are intended as a transparent reproducible implementation, not a
claim of equivalence to every published or commercial IGT variant. Record and
report the complete task configuration when comparing results.

## Schedule Assignment

- `shared`: subjects receive the same hidden payoff sequences, while visible
  labels remain counterbalanced.
- `per_subject`: payoff sequences are resolved from subject-specific seeds.

Every subject schedule is persisted before execution, including gains,
penalties, visible-label mapping, seed, and schedule ID.

## Interpretation

The standard summary reports advantageous choices minus disadvantageous
choices overall and in configurable trial blocks. It also reports final
balance, net earnings, total gains and penalties, action counts, invalid
attempts, and provider failures.

These are simulated model behaviors. API latency is not interpreted as human
reaction time, and pretrained task knowledge remains a possible validity
threat even with task-name and label masking.

## Sources

- [PsyToolkit Iowa Gambling Task](https://www.psytoolkit.org/library/igt.html)
- [Ahn et al., payoff distributions and bandit framing](https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/4/19514/files/2015/11/Ahn2008-1ex6c28.pdf)
