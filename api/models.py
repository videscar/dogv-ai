from sqlalchemy import (
    Column,
    Integer,
    String,
    Date,
    Text,
    JSON,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .db import Base


class DogvIssue(Base):
    __tablename__ = "dogv_issues"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    numero = Column(String, index=True, nullable=True)
    language = Column(String(5), nullable=False)
    title = Column(String, nullable=True)
    raw_json = Column(JSON, nullable=False)

    documents = relationship("DogvDocument", back_populates="issue")

    __table_args__ = (
        UniqueConstraint("date", "language", name="uq_issue_date_lang"),
    )


class DogvDocument(Base):
    __tablename__ = "dogv_documents"

    id = Column(Integer, primary_key=True, index=True)
    issue_id = Column(Integer, ForeignKey("dogv_issues.id"), index=True, nullable=False)

    section = Column(String, index=True, nullable=True)
    ref = Column(String, index=True, nullable=True)
    conselleria = Column(String, index=True, nullable=True)
    title = Column(String, nullable=True)
    type = Column(String, index=True, nullable=True)

    pdf_url = Column(String, nullable=True)
    html_url = Column(String, nullable=True)

    text = Column(Text, nullable=True)
    raw_json = Column(JSON, nullable=True)

    issue = relationship("DogvIssue", back_populates="documents")
