"""Add ensemble_configs table

Revision ID: 2f0f57cf6a2c
Revises: 478e112a9ef6
Create Date: 2026-04-13 11:18:40.763107

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2f0f57cf6a2c'
down_revision: Union[str, Sequence[str], None] = '478e112a9ef6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('ensemble_configs',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('rf_weight', sa.Float(), nullable=False),
    sa.Column('nn_weight', sa.Float(), nullable=False),
    sa.Column('catboost_weight', sa.Float(), nullable=False),
    sa.Column('is_active', sa.Boolean(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )

def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('ensemble_configs')
