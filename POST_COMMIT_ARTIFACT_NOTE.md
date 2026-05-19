# Round 7 Artifact Provenance Note

Round 7 paper artifacts in `results/round7_paper_artifacts/` were generated from a clean
pre-export code tree at commit:

```text
72f97912e294fc12470bbbf2862b503d0349ef58
```

The artifact files were committed in:

```text
72d4bc0bc6cbd99bbc6d838ca085b20b53860bd2
```

`artifact_manifest.json` records the clean pre-export commit because the artifact directory is
created after the clean-tree check and then committed as a follow-up artifact commit.

## Round 8

Round 8 paper artifacts in `results/round8_paper_artifacts/` were generated from a clean
pre-export code tree at commit:

```text
b0854cf9c00d8affd8a1bd3c8ffaca27b8a61162
```

The artifact files were committed in:

```text
1df4311a449ffcb5d96df7d0d40bf89f498e96e1
```

Round 8 explicitly marks decode cohort latency improvement as not claimable when the cohort
admission study shows p99 JCT regression beyond the 5% threshold; the committed claim is
traffic reduction, not latency improvement.
