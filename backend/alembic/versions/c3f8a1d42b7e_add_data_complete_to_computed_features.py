"""add data_complete to computed_features

Revision ID: c3f8a1d42b7e
Revises: 25a9c5ba24d6
Create Date: 2026-04-09 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c3f8a1d42b7e'
down_revision: Union[str, None] = '25a9c5ba24d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'computed_features',
        sa.Column('data_complete', sa.Boolean(), nullable=True, server_default=sa.text('1')),
    )


def downgrade() -> None:
    op.drop_column('computed_features', 'data_complete')
