"""add heartbeat_at and training_stage to experiments

Revision ID: a1b2c3d4e5f6
Revises: c3f8a1d42b7e
Create Date: 2026-04-10 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = 'c3f8a1d42b7e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    existing_cols = {c['name'] for c in inspector.get_columns('experiments')}

    if 'heartbeat_at' not in existing_cols:
        op.add_column(
            'experiments',
            sa.Column('heartbeat_at', sa.DateTime(), nullable=True),
        )
    if 'training_stage' not in existing_cols:
        op.add_column(
            'experiments',
            sa.Column('training_stage', sa.String(), nullable=True),
        )


def downgrade() -> None:
    with op.batch_alter_table('experiments') as batch_op:
        batch_op.drop_column('training_stage')
        batch_op.drop_column('heartbeat_at')
