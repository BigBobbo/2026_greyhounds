"""Add commission/min-odds/daily-exposure fields to bankroll_config.

The canonical staking module (ml/staking.py) is driven entirely by
BankrollConfig; these three fields complete it: exchange commission on net
winnings, a minimum acceptable price, and a whole-day exposure cap that the
portfolio allocator enforces across all recommendations.

Revision ID: i9d0e1f2g3h4
Revises: h8c9d0e1f2g3
Create Date: 2026-07-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "i9d0e1f2g3h4"
down_revision: Union[str, None] = "h8c9d0e1f2g3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "bankroll_config",
        sa.Column("commission_rate", sa.Float(), nullable=False,
                  server_default="0.05"),
    )
    op.add_column(
        "bankroll_config",
        sa.Column("min_odds", sa.Float(), nullable=False,
                  server_default="1.5"),
    )
    op.add_column(
        "bankroll_config",
        sa.Column("max_daily_exposure_pct", sa.Float(), nullable=False,
                  server_default="0.10"),
    )


def downgrade() -> None:
    op.drop_column("bankroll_config", "max_daily_exposure_pct")
    op.drop_column("bankroll_config", "min_odds")
    op.drop_column("bankroll_config", "commission_rate")
