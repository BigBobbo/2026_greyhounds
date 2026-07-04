"""add race_entries.scratched and unique index on dogs.gri_id

Revision ID: j0e1f2a3b4c5
Revises: i9d0e1f2a3b4
Create Date: 2026-06-11 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'j0e1f2a3b4c5'
down_revision: Union[str, None] = 'i9d0e1f2a3b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Audit tasks E5/E6.

    - race_entries.scratched: nullable Boolean flag for carded entries whose
      trap is absent from the results (dog did not run). Plain add_column —
      no constraint, so SQLite needs no batch_alter_table.
    - dogs.gri_id: unique index so two dogs can never share a GRI id.
      SQLite unique indexes permit multiple NULLs, so legacy name-only dogs
      (gri_id IS NULL) are unaffected. Fresh databases already get a
      table-level UNIQUE constraint from the initial schema migration, so
      the index is only created when no unique guarantee exists yet.
    """
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    if 'race_entries' in existing_tables:
        entry_cols = {c['name'] for c in inspector.get_columns('race_entries')}
        if 'scratched' not in entry_cols:
            op.add_column(
                'race_entries',
                sa.Column('scratched', sa.Boolean(), nullable=True),
            )

    if 'dogs' in existing_tables:
        already_unique = False
        for ix in inspector.get_indexes('dogs'):
            if ix.get('column_names') == ['gri_id'] and ix.get('unique'):
                already_unique = True
        try:
            for uc in inspector.get_unique_constraints('dogs'):
                if uc.get('column_names') == ['gri_id']:
                    already_unique = True
        except NotImplementedError:
            pass
        if not already_unique:
            op.create_index('ux_dogs_gri_id', 'dogs', ['gri_id'], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    existing_indexes = {
        ix['name'] for ix in inspector.get_indexes('dogs') if ix.get('name')
    }
    if 'ux_dogs_gri_id' in existing_indexes:
        op.drop_index('ux_dogs_gri_id', 'dogs')

    with op.batch_alter_table('race_entries') as batch_op:
        batch_op.drop_column('scratched')
