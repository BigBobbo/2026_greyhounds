from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, ForeignKey, UniqueConstraint
from app.database import Base


class ComputedFeature(Base):
    __tablename__ = "computed_features"

    id = Column(Integer, primary_key=True, index=True)
    race_entry_id = Column(Integer, ForeignKey("race_entries.id"), nullable=False, index=True)
    feature_def_id = Column(Integer, ForeignKey("feature_definitions.id"), nullable=False)
    value = Column(Float)
    computed_at = Column(DateTime, default=datetime.utcnow)
    # False when the dog's history may be incomplete (e.g. some tracks not
    # yet scraped for the feature's date window).  Downstream consumers can
    # filter these out to avoid training on potentially wrong values.
    data_complete = Column(Boolean, default=True)
    # Links this computed value to a named snapshot.  NULL means "current /
    # unversioned" (legacy behaviour: upserted in place).  When a version_id
    # is set, the row belongs to that snapshot and won't be overwritten.
    version_id = Column(Integer, ForeignKey("feature_versions.id"), nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint(
            "race_entry_id", "feature_def_id", "version_id",
            name="uq_computed_entry_feature_version",
        ),
    )
