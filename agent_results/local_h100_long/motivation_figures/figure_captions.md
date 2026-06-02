Figure 1: Local H100 traces show agent workflows as explicit state machines with typed prompt segments, not full-history transcripts.
Figure 2: Local H100 traces quantify delayed high-value state reuse and LRU-regret tokens by state type.
Figure 3: Local H100 serving evidence shows concurrency effects on request latency, workflow wall-clock time, and measured GPU memory; no provider-internal KV eviction is claimed.
Figure 4: BusyBarn-backed wafer replay shows finite-memory residency pressure through effective prefill and cache-byte misses under LRU.
Figure 5: BusyBarn-backed wafer replay compares LRU-system, KVFlow-like, ASG-online, and ASG-oracle policies on the opportunity workload.
Figure 6: Opportunity/control replay separates delayed-reuse-heavy traces from recency-control traces.
Figure 7: BusyBarn-backed wafer replay reports mapper/prefetch communication metrics when available.
