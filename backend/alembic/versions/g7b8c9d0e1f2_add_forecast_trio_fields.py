"""add forecast/trio fields to predictions and bet_records

Revision ID: g7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-05-04 09:00:00.000000

Adds the per-dog place/show probabilities and a JSON cache of the top
forecast/trio combos to the predictions table, and extends bet_records
with a bet_type discriminator + legs payload so forecast/trio bets can
be tracked through the bankroll alongside straight win bets.

The new fields are nullable and additive — every existing row keeps
working without backfill.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g7b8c9d0e1f2"
down_revision: Union[str, None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


PREDICTION_COLUMNS = [
    # Per-dog probabilities derived from the ordering simulator.
    ("place_probability", sa.Float(), True),  # P(dog finishes 1st or 2nd)
    ("show_probability", sa.Float(), True),   # P(dog finishes 1st, 2nd, or 3rd)
    # JSON cache of the top forecast / trio combos for the race this
    # entry belongs to. Stored on every row in the race so a single-row
    # API fetch can render the combos panel without a second query.
    ("forecast_combos_json", sa.Text(), True),
    ("trio_combos_json", sa.Text(), True),
]


BET_RECORD_COLUMNS = [
    # "win" (default — old rows), "place", "show", "forecast", "trio".
    ("bet_type", sa.String(), True),
    # JSON-encoded ordered list of race_entry_ids for multi-leg bets.
    # Single-dog bets leave it null and rely on race_entry_id.
    ("legs_json", sa.Text(), True),
    # Combo probability the model assigned at bet time (the Kelly p).
    ("combo_probability", sa.Float(), True),
]


def upgrade() -> None:
    from sqlalchemy import inspect

    bind = op.get_bind()
    inspector = inspect(bind)
    table_names = set(inspector.get_table_names())

    if "predictions" in table_names:
        existing = {c["name"] for c in inspector.get_columns("predictions")}
        for name, col_type, nullable in PREDICTION_COLUMNS:
            if name not in existing:
                op.add_column(
                    "predictions",
                    sa.Column(name, col_type, nullable=nullable),
                )

    if "bet_records" in table_names:
        existing = {c["name"] for c in inspector.get_columns("bet_records")}
        for name, col_type, nullable in BET_RECORD_COLUMNS:
            if name not in existing:
                op.add_column(
                    "bet_records",
                    sa.Column(name, col_type, nullable=nullable),
                )
        # Backfill bet_type='win' for legacy rows so summary queries can
        # filter cleanly without coalesce gymnastics.
        op.execute(
            "UPDATE bet_records SET bet_type='win' WHERE bet_type IS NULL"
        )


def downgrade() -> None:
    with op.batch_alter_table("bet_records") as batch_op:
        for name, _t, _n in reversed(BET_RECORD_COLUMNS):
            batch_op.drop_column(name)
    with op.batch_alter_table("predictions") as batch_op:
        for name, _t, _n in reversed(PREDICTION_COLUMNS):
            batch_op.drop_column(name)
