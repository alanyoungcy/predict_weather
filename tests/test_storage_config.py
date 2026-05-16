from wt.config import AppSettings


def test_settings_accept_vercel_supabase_and_mongo_aliases() -> None:
    settings = AppSettings.model_validate(
        {
            "POSTGRES_URL": "postgresql://user:pass@host/db",
            "SUPABASE_SECRET_KEY": "secret",
            "SUPABASE_URL": "https://example.supabase.co",
            "MONGODB_URI": "mongodb+srv://example",
            "MOTHERDUCK_TOKEN": "token",
        }
    )

    assert settings.supabase_db_url == "postgresql://user:pass@host/db"
    assert settings.supabase_secret_key == "secret"
    assert settings.mongodb_uri == "mongodb+srv://example"
    assert settings.motherduck_token == "token"
