# weather-trader

Weather prediction and market signal generation pipeline for US station-based
daily temperature and precipitation contracts across multiple venues.

## Status

Milestone `M1` is scaffolded:
- Python project layout under `src/wt`
- Configuration files under `config/`
- Environment template in `.env.example`
- DST-aware settlement-day window utility in `wt.utils.time`
- Exchange adapters scaffolded for Kalshi and Polymarket

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
worker, not inside Vercel functions. See
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
