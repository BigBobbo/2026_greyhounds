"""add place/show and position-distribution columns to predictions

Revision ID: e7f8a9b0c1d2
Revises: d4e5f6a7b8c9
Create Date: 2026-04-24 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e7f8a9b0c1d2'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    existing_cols = {c['name'] for c in inspector.get_columns('predictions')}

    if 'place2_probability' not in existing_cols:
        op.add_column(
            'predictions',
            sa.Column('place2_probability', sa.Float(), nullable=True),
        )
    if 'place3_probability' not in existing_cols:
        op.add_column(
            'predictions',
            sa.Column('place3_probability', sa.Float(), nullable=True),
        )
    if 'position_probs_json' not in existing_cols:
        op.add_column(
            'predictions',
            sa.Column('position_probs_json', sa.JSON(), nullable=True),
        )


def downgrade() -> None:
    op.drop_column('predictions', 'position_probs_json')
    op.drop_column('predictions', 'place3_probability')
    op.drop_column('predictions', 'place2_probability')
