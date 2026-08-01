"""Add track_weather table (daily per-track weather from Open-Meteo).

Revision ID: k1f2g3h4i5j6
Revises: j0e1f2g3h4i5
Create Date: 2026-07-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "k1f2g3h4i5j6"
down_revision: Union[str, None] = "j0e1f2g3h4i5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "track_weather",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("track_id", sa.Integer(), sa.ForeignKey("tracks.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("precip_mm", sa.Float(), nullable=True),
        sa.Column("temp_mean_c", sa.Float(), nullable=True),
        sa.Column("wind_max_kmh", sa.Float(), nullable=True),
        sa.Column("precip_prev48h_mm", sa.Float(), nullable=True),
        sa.UniqueConstraint("track_id", "date", name="uq_track_weather_day"),
    )
    op.create_index("ix_track_weather_track_id", "track_weather", ["track_id"])
    op.create_index("ix_track_weather_date", "track_weather", ["date"])


def downgrade() -> None:
    op.drop_table("track_weather")
