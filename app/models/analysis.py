
from sqlalchemy import Column, Integer, String, Float, Text

from app.database import Base


class Analysis(Base):

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(String, nullable=False)

    prediction = Column(String, nullable=False)

    confidence = Column(Float, nullable=False)

    result_json = Column(Text, nullable=False)
