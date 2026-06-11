"""create bankroll_config and bet_records via migration

These tables previously existed only because seed scripts called
Base.metadata.create_all() at boot — `alembic upgrade head` alone produced a
schema without them (and migration g7b8c9d0e1f2 skips its bet_records
column-adds when the table is absent). This catch-up migration makes Alembic
the single schema authority; the inspect() guards make it a no-op on
production databases that already converged via create_all().

Revision ID: i9d0e1f2a3b4
Revises: h8c9d0e1f2g3
Create Date: 2026-06-11 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'i9d0e1f2a3b4'
down_revision: Union[str, None] = 'h8c9d0e1f2g3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = inspector.get_table_names()

    if 'bankroll_config' not in existing_tables:
        op.create_table(
            'bankroll_config',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('initial_bankroll', sa.Float(), nullable=False),
            sa.Column('current_bankroll', sa.Float(), nullable=False),
            sa.Column('kelly_fraction', sa.Float(), nullable=False),
            sa.Column('min_edge', sa.Float(), nullable=False),
            sa.Column('max_stake_pct', sa.Float(), nullable=False),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.Column('updated_at', sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(op.f('ix_bankroll_config_id'), 'bankroll_config', ['id'], unique=False)

    if 'bet_records' not in existing_tables:
        op.create_table(
            'bet_records',
            sa.Column('id', sa.Integer(), nullable=False),
            sa.Column('race_entry_id', sa.Integer(), nullable=False),
            sa.Column('experiment_id', sa.Integer(), nullable=False),
            sa.Column('dog_name', sa.String(), nullable=True),
            sa.Column('track_name', sa.String(), nullable=True),
            sa.Column('race_date', sa.String(), nullable=True),
            sa.Column('race_number', sa.Integer(), nullable=True),
            sa.Column('trap', sa.Integer(), nullable=True),
            sa.Column('grade', sa.String(), nullable=True),
            sa.Column('bet_type', sa.String(), nullable=True),
            sa.Column('legs_json', sa.Text(), nullable=True),
            sa.Column('win_probability', sa.Float(), nullable=True),
            sa.Column('combo_probability', sa.Float(), nullable=True),
            sa.Column('odds_decimal', sa.Float(), nullable=True),
            sa.Column('implied_prob', sa.Float(), nullable=True),
            sa.Column('edge', sa.Float(), nullable=True),
            sa.Column('confidence_tier', sa.String(), nullable=True),
            sa.Column('stake', sa.Float(), nullable=False),
            sa.Column('stake_method', sa.String(), nullable=True),
            sa.Column('bankroll_before', sa.Float(), nullable=True),
            sa.Column('outcome', sa.String(), nullable=True),
            sa.Column('profit', sa.Float(), nullable=True),
            sa.Column('bankroll_after', sa.Float(), nullable=True),
            sa.Column('settled_at', sa.DateTime(), nullable=True),
            sa.Column('created_at', sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(['race_entry_id'], ['race_entries.id']),
            sa.ForeignKeyConstraint(['experiment_id'], ['experiments.id']),
            sa.PrimaryKeyConstraint('id'),
        )
        op.create_index(op.f('ix_bet_records_id'), 'bet_records', ['id'], unique=False)
        op.create_index(
            op.f('ix_bet_records_race_entry_id'), 'bet_records', ['race_entry_id'], unique=False
        )
        op.create_index(
            op.f('ix_bet_records_experiment_id'), 'bet_records', ['experiment_id'], unique=False
        )


def downgrade() -> None:
    op.drop_table('bet_records')
    op.drop_table('bankroll_config')
