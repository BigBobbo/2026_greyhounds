"""add missing indexes for query performance

Revision ID: d4e5f6a7b8c9
Revises: b2c3d4e5f6g7
Create Date: 2026-04-13 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = 'b2c3d4e5f6g7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # HIGH PRIORITY: columns used in nearly every training/prediction query
    op.create_index('ix_races_status', 'races', ['status'])
    op.create_index('ix_race_entries_finish_position', 'race_entries', ['finish_position'])
    op.create_index('ix_computed_features_feature_def_id', 'computed_features', ['feature_def_id'])
    op.create_index('ix_odds_snapshots_dog_id', 'odds_snapshots', ['dog_id'])
    op.create_index('ix_dogs_sire', 'dogs', ['sire'])

    # COMPOSITE: covers the most common GROUP BY patterns in race_features batch queries
    op.create_index(
        'ix_races_status_track_distance',
        'races',
        ['status', 'track_id', 'distance_m'],
    )
    op.create_index(
        'ix_computed_features_def_entry',
        'computed_features',
        ['feature_def_id', 'race_entry_id'],
    )


def downgrade() -> None:
    op.drop_index('ix_computed_features_def_entry', 'computed_features')
    op.drop_index('ix_races_status_track_distance', 'races')
    op.drop_index('ix_dogs_sire', 'dogs')
    op.drop_index('ix_odds_snapshots_dog_id', 'odds_snapshots')
    op.drop_index('ix_computed_features_feature_def_id', 'computed_features')
    op.drop_index('ix_race_entries_finish_position', 'race_entries')
    op.drop_index('ix_races_status', 'races')
