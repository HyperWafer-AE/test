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
  replicas only when the static replication-gain estimate is positive after a
  simple per-region SRAM pressure penalty.

Validation:

```bash
cd ../..
python -m compileall src models/agentic_krd_static tests
pytest tests/test_agent_krd.py
```

Run the default three-policy experiment:

```bash
cd models/agentic_krd_static
bash run.sh
```

Run a quick deterministic sweep:

```bash
python sweep.py --quick
```

Expected outputs are `results/central.json`, `results/full_replication.json`,
`results/krd_selective.json`, `results/summary.csv`, and
`results/sweep_summary.csv`.

Key metrics:

- `resident_bytes`: physical KV bytes resident across all placements.
- `unique_state_bytes`: one-copy footprint of all KV states.
- `extra_replica_bytes`: `resident_bytes - unique_state_bytes`.
- `capacity_violations`: number of KRD regions whose resident bytes exceed
  `len(region) * sram_capacity_bytes`.
- `kv_hop_bytes` and `communication_distances`: BusyBarn routing metrics from
  the existing `build_event()` backend.
