"""add scrape timestamps to races, race_entries and scrape_logs

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-04-27 19:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add scrape attribution timestamps.

    The `last_scrape_log_id` columns are intentionally added as plain Integer
    rather than ForeignKey columns: SQLite cannot ALTER a table to add a FK
    constraint, so we keep the FK as an ORM-level relationship only. This
    matches the convention used by earlier migrations in this project.
    """
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if 'scrape_logs' in existing_tables:
        scrape_log_cols = {c['name'] for c in inspector.get_columns('scrape_logs')}
        if 'heartbeat_at' not in scrape_log_cols:
            op.add_column(
                'scrape_logs',
                sa.Column('heartbeat_at', sa.DateTime(), nullable=True),
            )

    if 'races' in existing_tables:
        race_cols = {c['name'] for c in inspector.get_columns('races')}
        if 'last_scraped_at' not in race_cols:
            op.add_column(
                'races',
                sa.Column('last_scraped_at', sa.DateTime(), nullable=True),
            )
        if 'last_scrape_log_id' not in race_cols:
            op.add_column(
                'races',
                sa.Column('last_scrape_log_id', sa.Integer(), nullable=True),
            )
            existing_indexes = {
                ix['name'] for ix in inspector.get_indexes('races') if ix.get('name')
            }
            if 'ix_races_last_scrape_log_id' not in existing_indexes:
                op.create_index(
                    'ix_races_last_scrape_log_id', 'races', ['last_scrape_log_id']
                )

    if 'race_entries' in existing_tables:
        entry_cols = {c['name'] for c in inspector.get_columns('race_entries')}
        if 'last_scraped_at' not in entry_cols:
            op.add_column(
                'race_entries',
                sa.Column('last_scraped_at', sa.DateTime(), nullable=True),
            )
        if 'last_scrape_log_id' not in entry_cols:
            op.add_column(
                'race_entries',
                sa.Column('last_scrape_log_id', sa.Integer(), nullable=True),
            )
            existing_indexes = {
                ix['name']
                for ix in inspector.get_indexes('race_entries')
                if ix.get('name')
            }
            if 'ix_race_entries_last_scrape_log_id' not in existing_indexes:
                op.create_index(
                    'ix_race_entries_last_scrape_log_id',
                    'race_entries',
                    ['last_scrape_log_id'],
                )


def downgrade() -> None:
    with op.batch_alter_table('race_entries') as batch_op:
        batch_op.drop_index('ix_race_entries_last_scrape_log_id')
        batch_op.drop_column('last_scrape_log_id')
        batch_op.drop_column('last_scraped_at')

    with op.batch_alter_table('races') as batch_op:
        batch_op.drop_index('ix_races_last_scrape_log_id')
        batch_op.drop_column('last_scrape_log_id')
        batch_op.drop_column('last_scraped_at')

    with op.batch_alter_table('scrape_logs') as batch_op:
        batch_op.drop_column('heartbeat_at')
