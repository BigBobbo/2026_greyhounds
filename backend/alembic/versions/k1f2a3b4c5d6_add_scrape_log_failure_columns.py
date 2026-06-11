"""add scrape_logs.race_date and scrape_logs.track_code

Revision ID: k1f2a3b4c5d6
Revises: j0e1f2a3b4c5
Create Date: 2026-06-11 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'k1f2a3b4c5d6'
down_revision: Union[str, None] = 'j0e1f2a3b4c5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Audit task E7: structured failure bookkeeping.

    One ScrapeLog row is written per failed (track, date) pair with these
    columns set, so retry tooling can re-scrape exactly the failed pairs
    instead of parsing error_message strings. Both nullable — legacy rows
    and multi-track/multi-date parent jobs leave them NULL. Plain
    add_column, no constraints, so SQLite needs no batch_alter_table.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if 'scrape_logs' in inspector.get_table_names():
        cols = {c['name'] for c in inspector.get_columns('scrape_logs')}
        if 'race_date' not in cols:
            op.add_column('scrape_logs', sa.Column('race_date', sa.Date(), nullable=True))
        if 'track_code' not in cols:
            op.add_column('scrape_logs', sa.Column('track_code', sa.String(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c['name'] for c in inspector.get_columns('scrape_logs')}
    with op.batch_alter_table('scrape_logs') as batch_op:
        if 'track_code' in cols:
            batch_op.drop_column('track_code')
        if 'race_date' in cols:
            batch_op.drop_column('race_date')
