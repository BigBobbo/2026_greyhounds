from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Greyhound Predictor"
    database_url: str = "sqlite:///./data/greyhound.db"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # Auth: when set, every /api route except /api/health requires this
    # value in the X-API-Key header. Empty disables auth (local dev only).
    api_key: str = ""

    # Scheduler: set ENABLE_SCHEDULER=false to skip starting the in-process
    # APScheduler (e.g. for one-off maintenance containers or local dev).
    enable_scheduler: bool = True

    # Scraping
    scrape_delay: float = 2.0  # seconds between requests
    gri_base_url: str = "https://www.grireland.ie"
    greyhound_data_base_url: str = "https://www.greyhound-data.com"

    # ML
    model_artifacts_dir: str = "./data/models"

    # Backups (S3-compatible storage; all empty = backups disabled)
    backup_s3_bucket: str = ""
    backup_s3_endpoint_url: str = ""  # e.g. https://<account>.r2.cloudflarestorage.com
    backup_s3_access_key: str = ""
    backup_s3_secret_key: str = ""
    backup_s3_region: str = "auto"
    backup_s3_prefix: str = "backups"
    backup_retention_daily: int = 7
    backup_retention_weekly: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
