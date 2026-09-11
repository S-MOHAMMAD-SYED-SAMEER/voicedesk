"""create call and calendar tables

Revision ID: 94468e1a40f1
Revises: 
Create Date: 2026-09-11 12:13:33.652299

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "94468e1a40f1"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Postgres enum types outlive the tables that use them, so they are created
# and dropped explicitly rather than inline — otherwise a downgrade leaves them
# behind and the next upgrade fails with "type already exists".
ENUMS = (
    sa.Enum("inbound", "outbound", name="call_direction"),
    sa.Enum(
        "booked",
        "rescheduled",
        "cancelled",
        "message_taken",
        "transferred",
        "abandoned",
        "failed",
        name="call_outcome",
    ),
    sa.Enum("caller", "agent", name="turn_role"),
    sa.Enum("booked", "cancelled", name="appointment_status"),
)

# Two overlapping bookings for one staff member must be impossible, and no
# amount of application-level checking can guarantee that under concurrency:
# two callers can both pass an availability check and both insert. The range is
# half-open so an appointment ending at 10:00 does not collide with one
# starting at 10:00, and the constraint is partial so a cancelled appointment
# stops holding its slot.
NO_DOUBLE_BOOKING = """
    ALTER TABLE appointments
    ADD CONSTRAINT ck_appointments_no_double_booking
    EXCLUDE USING gist (
        staff_id WITH =,
        tstzrange(starts_at, ends_at, '[)') WITH &&
    ) WHERE (status = 'booked')
"""


def upgrade() -> None:
    """The call record and the calendar it books into."""
    # Needed for the `staff_id WITH =` half of the exclusion constraint: GiST
    # does not handle plain equality without it. Trusted since Postgres 13, so
    # the database owner can install it without superuser.
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    bind = op.get_bind()
    for enum in ENUMS:
        enum.create(bind, checkfirst=True)

    op.create_table('business_hours',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('weekday', sa.SmallInteger(), nullable=False),
    sa.Column('opens_at', sa.Time(), nullable=False),
    sa.Column('closes_at', sa.Time(), nullable=False),
    sa.CheckConstraint('closes_at > opens_at', name=op.f('ck_business_hours_closes_after_opens')),
    sa.CheckConstraint('weekday BETWEEN 0 AND 6', name=op.f('ck_business_hours_weekday_in_range')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_business_hours'))
    )
    op.create_index(op.f('ix_business_hours_weekday'), 'business_hours', ['weekday'], unique=False)
    op.create_table('calls',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('direction', postgresql.ENUM('inbound', 'outbound', name='call_direction', create_type=False), server_default='inbound', nullable=False),
    sa.Column('from_number', sa.String(length=32), nullable=False),
    sa.Column('to_number', sa.String(length=32), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('ended_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('outcome', postgresql.ENUM('booked', 'rescheduled', 'cancelled', 'message_taken', 'transferred', 'abandoned', 'failed', name='call_outcome', create_type=False), nullable=True),
    sa.Column('total_cost_usd', sa.Numeric(precision=10, scale=6), nullable=True),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_calls'))
    )
    op.create_index(op.f('ix_calls_started_at'), 'calls', ['started_at'], unique=False)
    op.create_table('services',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('duration_minutes', sa.Integer(), nullable=False),
    sa.Column('staff_id', sa.String(length=64), nullable=False),
    sa.Column('active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.CheckConstraint('duration_minutes > 0', name=op.f('ck_services_duration_minutes_positive')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_services'))
    )
    op.create_index(op.f('ix_services_staff_id'), 'services', ['staff_id'], unique=False)
    op.create_table('appointments',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('customer_name', sa.String(length=128), nullable=False),
    sa.Column('phone', sa.String(length=32), nullable=False),
    sa.Column('service_id', sa.Uuid(), nullable=False),
    sa.Column('staff_id', sa.String(length=64), nullable=False),
    sa.Column('starts_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('ends_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('status', postgresql.ENUM('booked', 'cancelled', name='appointment_status', create_type=False), server_default='booked', nullable=False),
    sa.Column('call_id', sa.Uuid(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('ends_at > starts_at', name=op.f('ck_appointments_ends_after_starts')),
    sa.ForeignKeyConstraint(['call_id'], ['calls.id'], name=op.f('fk_appointments_call_id_calls'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['service_id'], ['services.id'], name=op.f('fk_appointments_service_id_services'), ondelete='RESTRICT'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_appointments'))
    )
    op.create_index(op.f('ix_appointments_call_id'), 'appointments', ['call_id'], unique=False)
    op.create_index(op.f('ix_appointments_service_id'), 'appointments', ['service_id'], unique=False)
    op.create_index('ix_appointments_staff_id_starts_at', 'appointments', ['staff_id', 'starts_at'], unique=False)
    op.create_table('turns',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('call_id', sa.Uuid(), nullable=False),
    sa.Column('role', postgresql.ENUM('caller', 'agent', name='turn_role', create_type=False), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('audio_ms', sa.Integer(), nullable=True),
    sa.Column('stt_latency_ms', sa.Integer(), nullable=True),
    sa.Column('llm_latency_ms', sa.Integer(), nullable=True),
    sa.Column('tts_latency_ms', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['call_id'], ['calls.id'], name=op.f('fk_turns_call_id_calls'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_turns'))
    )
    op.create_index(op.f('ix_turns_call_id'), 'turns', ['call_id'], unique=False)
    op.create_table('tool_calls',
    sa.Column('id', sa.Uuid(), server_default=sa.text('gen_random_uuid()'), nullable=False),
    sa.Column('turn_id', sa.Uuid(), nullable=False),
    sa.Column('tool_name', sa.String(length=64), nullable=False),
    sa.Column('arguments', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('success', sa.Boolean(), nullable=False),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('latency_ms', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['turn_id'], ['turns.id'], name=op.f('fk_tool_calls_turn_id_turns'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_tool_calls'))
    )
    op.create_index(op.f('ix_tool_calls_turn_id'), 'tool_calls', ['turn_id'], unique=False)

    # The database, not the booking code, is the authority on double booking.
    op.execute(NO_DOUBLE_BOOKING)


def downgrade() -> None:
    op.execute(
        "ALTER TABLE appointments DROP CONSTRAINT "
        "IF EXISTS ck_appointments_no_double_booking"
    )
    op.drop_index(op.f('ix_tool_calls_turn_id'), table_name='tool_calls')
    op.drop_table('tool_calls')
    op.drop_index(op.f('ix_turns_call_id'), table_name='turns')
    op.drop_table('turns')
    op.drop_index('ix_appointments_staff_id_starts_at', table_name='appointments')
    op.drop_index(op.f('ix_appointments_service_id'), table_name='appointments')
    op.drop_index(op.f('ix_appointments_call_id'), table_name='appointments')
    op.drop_table('appointments')
    op.drop_index(op.f('ix_services_staff_id'), table_name='services')
    op.drop_table('services')
    op.drop_index(op.f('ix_calls_started_at'), table_name='calls')
    op.drop_table('calls')
    op.drop_index(op.f('ix_business_hours_weekday'), table_name='business_hours')
    op.drop_table('business_hours')

    bind = op.get_bind()
    for enum in ENUMS:
        enum.drop(bind, checkfirst=True)
    # btree_gist is left installed: other things in the database may rely on
    # it, and dropping an extension a migration merely ensured is not this
    # migration's to undo.
