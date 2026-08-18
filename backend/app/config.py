from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    app_name: str = "Greyhound Predictor"
    database_url: str = "sqlite:///./data/greyhound.db"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "*"]

    # Scraping
    scrape_delay: float = 2.0  # seconds between requests
    gri_base_url: str = "https://www.grireland.ie"
    greyhound_data_base_url: str = "https://www.greyhound-data.com"

    # Betfair
    betfair_api_key: str = ""
    betfair_username: str = ""
    betfair_password: str = ""
    # Optional client-certificate pair. When both are set the client uses
    # Betfair's certificate login instead of the interactive one — the
    # only path that works on an account with two-factor authentication.
    betfair_cert_file: str = ""
    betfair_cert_key_file: str = ""

    # ML
    model_artifacts_dir: str = "./data/models"

    # Admin: bearer token protecting the database-backup download endpoint.
    # Empty string disables the endpoint entirely. Set a long random value
    # in the deployment environment to enable HTTPS backups (the hosting
    # volume has no other export path).
    admin_backup_token: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
