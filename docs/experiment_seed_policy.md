# Experiment Seed Policy

For comparable LION experiments, use the following fixed seed sequence unless
the user explicitly specifies a seed:

```text
3407 -> 42 -> 1024 -> 810
```

Selection rule:

- For the first run of a new experiment family, use `3407`.
- If the same experiment needs another seed, use `42`.
- Then use `1024`.
- Then use `810`.
- Four seeds are the default upper bound for routine robustness checks.

Rationale:

- `3407` and `42` are commonly used community seeds.
- `1024` is `2^10`.
- `810` is the user's birthday.

When launching a run, write the chosen seed into the run tag and pass it
explicitly through the training command, for example `--fix_random_seed --seed
3407` or the wrapper variables `FIX_RANDOM_SEED=1 RANDOM_SEED=3407`.
