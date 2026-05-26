# WaferStateFlow Dynamic Hotness Sweep

## Executive Summary

This sweep compares a static policy that cannot see runtime-hot candidate states with a dynamic policy that can promote them after observation.

| p(dynamic) | static latency | dynamic latency | materialization delta | H3 signal |
| ---: | ---: | ---: | ---: | --- |
| 0.00 | 63.288 | 63.288 | 0 | no |
| 0.25 | 63.288 | 63.288 | 0 | no |
| 0.50 | 63.288 | 63.288 | 0 | no |
| 0.75 | 63.288 | 63.288 | 0 | no |

## Failure Cases

When `p(dynamic)=0`, dynamic policy should not help. If it does, the simulator is over-crediting dynamic adaptation.

## What This Means for the Paper

H3 is not supported in this synthetic sweep. A real trace is still needed before claiming dynamic hotness as a core contribution.
