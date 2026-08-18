"""Add odds_snapshots.market_id / selection_id — the Betfair market and
runner identifiers a snapshot came from.

Without them a captured price is a dead end: settling the same market for
its Betfair Starting Price after the race means re-listing the exact
marketId we priced, and there is no way to re-derive it from (race, dog)
once the market has closed and dropped out of the catalogue.

Revision ID: l2g3h4i5j6k7
Revises: k1f2g3h4i5j6
Create Date: 2026-08-18
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "l2g3h4i5j6k7"
down_revision: Union[str, None] = "k1f2g3h4i5j6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "odds_snapshots", sa.Column("market_id", sa.String(), nullable=True),
    )
    op.add_column(
        "odds_snapshots", sa.Column("selection_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_odds_snapshots_market_id", "odds_snapshots", ["market_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_odds_snapshots_market_id", table_name="odds_snapshots")
    op.drop_column("odds_snapshots", "selection_id")
    op.drop_column("odds_snapshots", "market_id")
