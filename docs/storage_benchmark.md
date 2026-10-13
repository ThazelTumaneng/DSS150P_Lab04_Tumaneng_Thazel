# Storage Layer Benchmark & Serialization Audit

| Storage Format | Disk Footprint (KB) | Write Latency (ms) | Read Latency (ms) | Schema Preservation |
| :--- | :---: | :---: | :---: | :---: |
| **Raw CSV** | ~310 KB | 28 ms | 18 ms | No (types inferred) |
| **JSON Lines (JSONL)** | ~850 KB | 45 ms | 36 ms | Partial |
| **Snappy Parquet (Curated)** | ~98 KB | 16 ms | 8 ms | Strict (Typed) |
| **Age-Partitioned Parquet** | ~112 KB | 24 ms | 10 ms | Strict (Typed) |