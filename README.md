# weather-trader

Weather prediction and market signal generation pipeline for US station-based
daily temperature and precipitation contracts across multiple venues.

## Status

Implemented foundation:
- M1-M5 scaffolding and ground-truth/model-training foundations
- Probability distributions and bucket probability alignment
- Venue-agnostic EV and signal generation for Kalshi and Polymarket rows
- Read-only Kalshi and Polymarket market clients
- Live feature builder with Herbie primary ingestion and Open-Meteo/NWS fallbacks
- Worker entrypoints for morning/evening dry-runs, training, and basic backtests

## Quickstart

Use the `agentenv` conda environment for all commands.

```bash
/opt/miniconda3/envs/agentenv/bin/python -m wt.utils.time
/opt/miniconda3/envs/agentenv/bin/pytest -q
```

## Deployment Shape

This project is now being designed for:

- Vercel: dashboard and read API only
- Supabase: canonical structured tables
- MongoDB Atlas: raw market snapshots and nested payloads
- DuckDB or MotherDuck: analytics, backtests, and feature research

Heavy backfills, Herbie downloads, and model training should run on an external
worker, not inside Vercel functions. Live worker smoke test:

```bash
/opt/miniconda3/envs/agentenv/bin/python -m wt.orchestration.cron_evening --dry-run --station KNYC
```

See
[docs/deployment.md](/Volumes/Orico/code/pythoncode/polymarkettool/polymarket-weather/docs/deployment.md).

## Vercel Storage Integration

The config layer now accepts Vercel-managed integration variables directly:

- Supabase: `POSTGRES_URL`, `POSTGRES_URL_NON_POOLING`, `SUPABASE_URL`, `SUPABASE_SECRET_KEY`
- MongoDB Atlas: `MONGODB_URI`
- MotherDuck: `MOTHERDUCK_TOKEN`, `MOTHERDUCK_READONLY_TOKEN`

Local development flow:

```bash
vercel env pull .env.local
/opt/miniconda3/envs/agentenv/bin/python scripts/check_storage_connections.py
```

## Mongo Test Endpoint

There is a minimal Vercel-ready MongoDB connectivity endpoint at `api/app.py`.

- Route: `/api/mongo-test`
- Method: `GET`
- Optional protection: set `MONGO_TEST_TOKEN` and send it as `x-mongo-test-token` or `?token=...`

The endpoint returns sanitized diagnostics only: whether `MONGODB_URI` is present,
the resolved URI host, topology type, server types, and the ping result. It does
not echo secrets.
