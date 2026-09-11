"""add provider call sid

The carrier's own identifier for a call, so a VoiceDesk row and a telephony
record can be matched up afterwards — which is what a billing question, a
recording or a support ticket is keyed by.

Nullable, because a browser-harness call has no carrier and rows written
before this migration have no identifier to backfill; adding it to a table
with rows in it is therefore safe and needs no default.

Unique, because it is the database refusing a second row for one call. A
carrier retrying a webhook is ordinary, and a constraint is better protection
against a duplicate than application code remembering to look first.

Revision ID: fa76475b048b
Revises: 94468e1a40f1
Create Date: 2026-09-11 16:59:57.878986
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "fa76475b048b"
down_revision: str | None = "94468e1a40f1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "uq_calls_provider_call_sid"


def upgrade() -> None:
    op.add_column(
        "calls", sa.Column("provider_call_sid", sa.String(length=64), nullable=True)
    )
    op.create_unique_constraint(CONSTRAINT, "calls", ["provider_call_sid"])


def downgrade() -> None:
    op.drop_constraint(CONSTRAINT, "calls", type_="unique")
    op.drop_column("calls", "provider_call_sid")
