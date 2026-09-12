"""add call cost rows

What each call spent, one row per component: the model, recognition, synthesis
and the line itself. Usage is always recorded; `cost_usd` is nullable because
VoiceDesk ships no vendor prices, and an unpriced component must stay visibly
unpriced rather than be rounded down to free.

Two unique indexes, not one constraint. `(turn_id, component)` cannot police
the rows whose `turn_id` is null — PostgreSQL treats nulls in a unique index
as distinct, so a second telephony row for the same call would be accepted —
so a partial index on `(call_id, component) WHERE turn_id IS NULL` covers
exactly those. Between them, re-recording a component overwrites rather than
duplicates.

A new table only. Nothing existing is altered, so this is safe against a
populated database and needs no backfill: `calls.total_cost_usd` has been
nullable and unwritten since the first migration.

Revision ID: 11bc87a6af66
Revises: fa76475b048b
Create Date: 2026-09-12 04:42:53.043841
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "11bc87a6af66"
down_revision: str | None = "fa76475b048b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Created and dropped explicitly, for the same reason as the enums in the
# first migration: a Postgres enum type outlives the table that uses it, and
# an inline one left behind by a downgrade breaks the next upgrade.
COMPONENT = sa.Enum("llm", "stt", "tts", "telephony", name="cost_component")

TURN_COMPONENT_UNIQUE = "uq_call_costs_turn_component"
CALL_COMPONENT_UNIQUE = "uq_call_costs_call_component_without_turn"
CALL_COMPONENT_INDEX = "ix_call_costs_call_id_component"


def upgrade() -> None:
    """The per-component record of what a call consumed."""
    COMPONENT.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "call_costs",
        sa.Column(
            "id",
            sa.Uuid(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("call_id", sa.Uuid(), nullable=False),
        # Null for a component that belongs to the whole call rather than to
        # anything said during it.
        sa.Column("turn_id", sa.Uuid(), nullable=True),
        sa.Column(
            "component",
            postgresql.ENUM(
                "llm", "stt", "tts", "telephony",
                name="cost_component",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=True),
        # Measured, and therefore never null. `unit_type` says what they count.
        sa.Column("input_units", sa.Numeric(precision=18, scale=3), nullable=False),
        sa.Column(
            "output_units",
            sa.Numeric(precision=18, scale=3),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("unit_type", sa.String(length=32), nullable=False),
        # Per single unit, so that cost_usd = unit_price_usd * units is
        # checkable in SQL. Null for a component billed at two rates.
        sa.Column(
            "unit_price_usd", sa.Numeric(precision=18, scale=12), nullable=True
        ),
        # Null means "not priced". It does not mean free.
        sa.Column("cost_usd", sa.Numeric(precision=10, scale=6), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["call_id"],
            ["calls.id"],
            name=op.f("fk_call_costs_call_id_calls"),
            ondelete="CASCADE",
        ),
        # SET NULL, not CASCADE: if a turn is ever deleted, what it spent
        # still happened and the call still owns it.
        sa.ForeignKeyConstraint(
            ["turn_id"],
            ["turns.id"],
            name=op.f("fk_call_costs_turn_id_turns"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_call_costs")),
    )
    op.create_index(
        op.f("ix_call_costs_call_id"), "call_costs", ["call_id"], unique=False
    )
    op.create_index(
        CALL_COMPONENT_INDEX, "call_costs", ["call_id", "component"], unique=False
    )
    op.create_index(
        CALL_COMPONENT_UNIQUE,
        "call_costs",
        ["call_id", "component"],
        unique=True,
        postgresql_where=sa.text("turn_id IS NULL"),
    )
    op.create_index(
        TURN_COMPONENT_UNIQUE, "call_costs", ["turn_id", "component"], unique=True
    )


def downgrade() -> None:
    op.drop_index(TURN_COMPONENT_UNIQUE, table_name="call_costs")
    op.drop_index(
        CALL_COMPONENT_UNIQUE,
        table_name="call_costs",
        postgresql_where=sa.text("turn_id IS NULL"),
    )
    op.drop_index(CALL_COMPONENT_INDEX, table_name="call_costs")
    op.drop_index(op.f("ix_call_costs_call_id"), table_name="call_costs")
    op.drop_table("call_costs")

    COMPONENT.drop(op.get_bind(), checkfirst=True)
