"""expand predictions table with confidence, kelly, and edge metadata

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-05-03 12:00:00.000000

Persists the full per-prediction context (Kelly staking, confidence tier,
margin, entropy, edge, is_value, bankroll_used, data_completeness) so saved
predictions can be replayed in the UI without re-running the model.

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f6a7b8c9d0e1'
down_revision: Union[str, None] = 'e5f6a7b8c9d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


NEW_COLUMNS = [
    ('confidence_tier', sa.String(), True),
    ('margin', sa.Float(), True),
    ('entropy', sa.Float(), True),
    ('edge', sa.Float(), True),
    ('is_value', sa.Boolean(), True),
    ('kelly_bet', sa.Boolean(), True),
    ('kelly_stake', sa.Float(), True),
    ('kelly_stake_pct', sa.Float(), True),
    ('kelly_full_pct', sa.Float(), True),
    ('kelly_expected_value', sa.Float(), True),
    ('kelly_implied_prob', sa.Float(), True),
    ('kelly_reason', sa.String(), True),
    ('sp_decimal_at_pred', sa.Float(), True),
    ('data_completeness', sa.Float(), True),
    ('bankroll_used', sa.Float(), True),
    ('updated_at', sa.DateTime(), True),
]


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)

    if 'predictions' not in set(inspector.get_table_names()):
        return

    existing_cols = {c['name'] for c in inspector.get_columns('predictions')}
    for name, col_type, nullable in NEW_COLUMNS:
        if name not in existing_cols:
            op.add_column(
                'predictions',
                sa.Column(name, col_type, nullable=nullable),
            )


def downgrade() -> None:
    with op.batch_alter_table('predictions') as batch_op:
        for name, _col_type, _nullable in reversed(NEW_COLUMNS):
            batch_op.drop_column(name)
