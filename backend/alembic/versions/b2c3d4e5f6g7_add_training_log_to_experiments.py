"""add training_log to experiments

Revision ID: b2c3d4e5f6g7
Revises: a1b2c3d4e5f6
Create Date: 2026-04-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6g7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    existing_cols = {c['name'] for c in inspector.get_columns('experiments')}

    if 'training_log' not in existing_cols:
        op.add_column(
            'experiments',
            sa.Column('training_log', sa.Text(), nullable=True),
        )


def downgrade() -> None:
    with op.batch_alter_table('experiments') as batch_op:
        batch_op.drop_column('training_log')
