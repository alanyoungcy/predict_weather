# Deployment Architecture

## Recommended split

This repository should be designed as a split system:

- Vercel: Next.js UI, read API, authenticated admin views, lightweight refresh endpoints
- Supabase Postgres: canonical relational store for labels, predictions, signals, run summaries
- MongoDB Atlas: raw market snapshots, order books, API payload archives, debug traces
- DuckDB or MotherDuck: backtests, feature analysis, offline research, ad hoc analytics
- External scheduler/worker: historical backfill, live ingestion, training, calibration, housekeeping

## Why this split exists

The weather pipeline has three workloads with very different characteristics:

- Online serving: low-latency reads for dashboards and signal views
- Operational writes: append-heavy daily predictions and signals
- Heavy batch compute: CF6/GHCN ingestion, Herbie downloads, feature backfills, model training

Vercel is a poor fit for the heavy batch path, especially on Hobby. Treat it as the presentation and read-API layer, not the compute substrate for historical data rebuilds.

## Store responsibilities

### Supabase

Use for structured, queryable tables:

- `labels_daily`
- `features_live`
- `predictions_daily`
- `signals_daily`
- `run_summary`
- `drift_residuals`

Recommended posture:

- Row-level security off for service-role worker writes, on for dashboard reads if public access is exposed later
- Store compact columns only; do not archive large raw API payloads here
- Partition logically by station and target date at the query layer

### MongoDB Atlas

Use for unstructured or nested data:

- Kalshi/Polymarket market payloads
- Order books
- API error payloads
- Debug snapshots of upstream provider responses

Operational rule:

- Add TTL indexes to snapshot collections so old raw payloads self-delete
- Keep this store thin to avoid free-tier transfer and storage pressure

### DuckDB / MotherDuck

Use for:

- local model training datasets
- backtests
- reliability tables
- exploratory analysis

Default behavior:

- local DuckDB first for development
- MotherDuck optional when you want remote persistence or SQL sharing across machines

## Retention policy

Suggested defaults are encoded in `config/storage.yaml`.

- keep labels forever
- keep local raw downloads 30 days
- keep local interim artifacts 30 days
- keep local feature files 180 days
- keep local predictions and signals 365 days
- keep only the latest 4 model version directories locally
- keep raw market snapshots 30 days in MongoDB Atlas via TTL

## Data flow

1. External worker ingests CF6/GHCN/NWP/market data.
2. Worker writes canonical rows to Supabase.
3. Worker writes raw payload archives to MongoDB Atlas.
4. Worker optionally mirrors parquet artifacts locally.
5. Vercel reads from Supabase for dashboard/API responses.
6. Research and backtests use DuckDB or MotherDuck, not Vercel runtime storage.
