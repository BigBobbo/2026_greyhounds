"""Add race_entries.running_positions (per-marker positions from GRI
dog-profile form lines, e.g. "1222" = led at the first bend then held
second). Written by scraping/dog_profile_scraper.py.

Revision ID: j0e1f2g3h4i5
Revises: i9d0e1f2g3h4
Create Date: 2026-07-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "j0e1f2g3h4i5"
down_revision: Union[str, None] = "i9d0e1f2g3h4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "race_entries",
        sa.Column("running_positions", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("race_entries", "running_positions")
