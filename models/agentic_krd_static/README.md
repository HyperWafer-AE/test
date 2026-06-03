# WaferAgent Static KRD Placement

This model directory implements the first-stage WaferAgent prototype on top of
BusyBarn's existing tensor placement, `build_event()` communication inference,
BALD/XY routing, and analytical `event_driver()`.

It builds a synthetic coding-agent KV trace, materializes each KV block as a
`tensor_notation` split in tag namespace `(2, state_id)`, places shared and
private KV states with one of three policies, then runs a vectorunit `lookup`
microbenchmark for all agent KV reads.

Policies:

- `central`: shared KV states have one global placement; private states are at
  the owner agent decode node.
- `full_replication`: shared KV states are replicated at every KRD anchor.
- `krd_selective`: shared KV states start at the global medoid and add KRD
  replicas only when the static replication-gain estimate is positive.

Run:

```bash
bash run.sh
```

Expected outputs are `results/central.json`, `results/full_replication.json`,
`results/krd_selective.json`, and `results/summary.csv`.

