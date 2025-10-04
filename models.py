from sqlalchemy import Column, String, Float
from database import Base

class Device(Base):
    __tablename__ = "devices"

    # ♡ Primary identifiers
    did = Column(String, primary_key=True, index=True)
    device_name = Column(String, unique=True, index=True)

    # ♡ Kyber PQC keys
    kyber_pk = Column(String)   # Public key
    kyber_sk = Column(String)   # Secret key

    # ♡ Last known telemetry fields (optional, can be null initially)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    altitude = Column(Float, nullable=True)
