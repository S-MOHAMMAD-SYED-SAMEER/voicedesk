"""add staff requests

Messages left for the team, requests that somebody call a caller back (there is
no live transfer), and the record that the system could not answer. One new
table; nothing existing is altered, so it is safe against a populated database.

`call_id` is SET NULL on delete, like `appointments.call_id`: a request that
somebody still has to act on must outlive the retention of the call it came
from.

Revision ID: 5d1c7a2e9b34
Revises: 11bc87a6af66
Create Date: 2026-10-22 09:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "5d1c7a2e9b34"
down_revision: str | Sequence[str] | None = "11bc87a6af66"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "staff_requests",
        sa.Column(
            "id", sa.Uuid(), server_default=sa.text("gen_random_uuid()"), nullable=False
        ),
        sa.Column("call_id", sa.Uuid(), nullable=True),
        sa.Column(
            "kind",
            sa.Enum("message", "transfer", "failure", name="staff_request_kind"),
            nullable=False,
        ),
        sa.Column("caller_name", sa.String(length=128), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("handled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["call_id"],
            ["calls.id"],
            name=op.f("fk_staff_requests_call_id_calls"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_staff_requests")),
    )
    op.create_index(
        op.f("ix_staff_requests_call_id"), "staff_requests", ["call_id"], unique=False
    )
    op.create_index(
        op.f("ix_staff_requests_created_at"),
        "staff_requests",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_staff_requests_created_at"), table_name="staff_requests")
    op.drop_index(op.f("ix_staff_requests_call_id"), table_name="staff_requests")
    op.drop_table("staff_requests")
    sa.Enum(name="staff_request_kind").drop(op.get_bind(), checkfirst=True)
