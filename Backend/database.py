from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "postgresq+asyncpg://a:newpassword:password@localhost:5432l/marketing_campaigns"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DBCampaignInput(Base):
    __tablename__ = "campaign_inputs"
    
    id = Column(Integer, primary_key=True, index=True)
    campaign_type = Column(String)
    customer_segment = Column(String)
    location = Column(String)
    language = Column(String)
    duration_days = Column(Integer)
    month = Column(Integer)
    year = Column(Integer)
    conversion_weight = Column(Float)
    roi_weight = Column(Float)
    engagement_weight = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)