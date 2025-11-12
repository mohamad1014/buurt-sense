"""Generic Alembic revision script."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

def upgrade():
    """Upgrade database schema."""
    ${upgrades if upgrades else 'pass'}

def downgrade():
    """Downgrade database schema."""
    ${downgrades if downgrades else 'pass'}
