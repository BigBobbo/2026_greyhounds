"""add data_complete and feature versioning to computed_features

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
    # Feature versions table
    op.create_table(
        'feature_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('coverage_snapshot', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )
    op.create_index(op.f('ix_feature_versions_id'), 'feature_versions', ['id'], unique=False)

    # New columns on computed_features
    op.add_column(
        'computed_features',
        sa.Column('data_complete', sa.Boolean(), nullable=True, server_default=sa.text('1')),
    )
    op.add_column(
        'computed_features',
        sa.Column('version_id', sa.Integer(), sa.ForeignKey('feature_versions.id'), nullable=True),
    )
    op.create_index('ix_computed_features_version_id', 'computed_features', ['version_id'])

    # Replace the old unique constraint with one that includes version_id.
    # SQLite doesn't support DROP CONSTRAINT, so we use batch mode.
    with op.batch_alter_table('computed_features') as batch_op:
        batch_op.drop_constraint('uq_computed_entry_feature', type_='unique')
        batch_op.create_unique_constraint(
            'uq_computed_entry_feature_version',
            ['race_entry_id', 'feature_def_id', 'version_id'],
        )


def downgrade() -> None:
    with op.batch_alter_table('computed_features') as batch_op:
        batch_op.drop_constraint('uq_computed_entry_feature_version', type_='unique')
        batch_op.create_unique_constraint(
            'uq_computed_entry_feature',
            ['race_entry_id', 'feature_def_id'],
        )
    op.drop_index('ix_computed_features_version_id', table_name='computed_features')
    op.drop_column('computed_features', 'version_id')
    op.drop_column('computed_features', 'data_complete')
    op.drop_index(op.f('ix_feature_versions_id'), table_name='feature_versions')
    op.drop_table('feature_versions')
