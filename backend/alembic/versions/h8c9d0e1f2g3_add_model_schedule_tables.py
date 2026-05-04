"""add model_schedule and scheduled_prediction_run tables

Revision ID: h8c9d0e1f2g3
Revises: g7b8c9d0e1f2
Create Date: 2026-05-04 12:00:00.000000

Introduces two tables that support the daily scheduled prediction tracker:

- ``model_schedule`` — one row per experiment the user wants to run on a
  schedule. Holds the cron time, a per-model timezone, an ``enabled`` flag,
  and ``is_main`` (only one row may be true). Bankroll fields are deferred
  to a later migration.

- ``scheduled_prediction_run`` — append-only audit log of each automated
  job run. Lets the UI show "last run", "next run", and surface failures
  without scraping APScheduler internal state.

The migration is additive and uses inspect() to stay idempotent so it can
re-run safely on partially-migrated environments (matches the pattern used
by g7b8c9d0e1f2_add_forecast_trio_fields).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "h8c9d0e1f2g3"
down_revision: Union[str, None] = "g7b8c9d0e1f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    table_names = set(inspector.get_table_names())

    if "model_schedule" not in table_names:
        op.create_table(
            "model_schedule",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "experiment_id",
                sa.Integer,
                sa.ForeignKey("experiments.id", ondelete="CASCADE"),
                nullable=False,
                unique=True,
                index=True,
            ),
            sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.true()),
            sa.Column("is_main", sa.Boolean, nullable=False, server_default=sa.false()),
            sa.Column("cron_hour", sa.Integer, nullable=False, server_default="8"),
            sa.Column("cron_minute", sa.Integer, nullable=False, server_default="30"),
            sa.Column(
                "timezone",
                sa.String,
                nullable=False,
                server_default="Europe/Dublin",
            ),
            # If true, the daily job will scrape upcoming cards before predicting
            # so missing meetings get filled in automatically.
            sa.Column(
                "scrape_upcoming",
                sa.Boolean,
                nullable=False,
                server_default=sa.true(),
            ),
            # How many days ahead of "today" the job should predict for.
            # Most cards are published the evening before, so 1 is the
            # sensible default (today + tomorrow gets covered between
            # consecutive runs).
            sa.Column(
                "predict_days_ahead",
                sa.Integer,
                nullable=False,
                server_default="1",
            ),
            sa.Column("created_at", sa.DateTime, nullable=False),
            sa.Column("updated_at", sa.DateTime, nullable=False),
        )

    if "scheduled_prediction_run" not in table_names:
        op.create_table(
            "scheduled_prediction_run",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column(
                "model_schedule_id",
                sa.Integer,
                sa.ForeignKey("model_schedule.id", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            sa.Column("run_date", sa.Date, nullable=False, index=True),
            # "running" | "success" | "partial" | "failed"
            sa.Column("status", sa.String, nullable=False),
            # "scheduled" (cron-fired) or "manual" (user-triggered via API)
            sa.Column("trigger", sa.String, nullable=False, server_default="scheduled"),
            sa.Column("races_predicted", sa.Integer, nullable=False, server_default="0"),
            sa.Column("races_skipped", sa.Integer, nullable=False, server_default="0"),
            sa.Column("predictions_written", sa.Integer, nullable=False, server_default="0"),
            sa.Column("error_message", sa.Text, nullable=True),
            sa.Column("started_at", sa.DateTime, nullable=False),
            sa.Column("finished_at", sa.DateTime, nullable=True),
        )
        op.create_index(
            "ix_scheduled_run_schedule_date",
            "scheduled_prediction_run",
            ["model_schedule_id", "run_date"],
        )


def downgrade() -> None:
    op.drop_index("ix_scheduled_run_schedule_date", table_name="scheduled_prediction_run")
    op.drop_table("scheduled_prediction_run")
    op.drop_table("model_schedule")
