from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import joblib
import pandas as pd
from datetime import datetime
from typing import List
import traceback
import logging
import sys
import numpy as np
import os
import time
from dotenv import load_dotenv
from fastapi.concurrency import run_in_threadpool

# Load environment variables
load_dotenv()

# Configure logging to handle Unicode properly
class UnicodeEscapeHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            msg = msg.encode('ascii', 'replace').decode('ascii')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        UnicodeEscapeHandler(sys.stdout),
        logging.FileHandler("api_debug.log", encoding='utf-8')
    ],
)

# Database setup with connection pool optimization
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://a:newpassword@localhost:5432/marketing_campaigns")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)
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

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Load Pre-Trained ML Models and Preprocessor
try:
    models = {
        'conversion_rate': joblib.load('marketing_campaign_model_conversion_rate.pkl'),
        'roi': joblib.load('marketing_campaign_model_roi.pkl'),
        'engagement': joblib.load('marketing_campaign_model_engagement.pkl')
    }
    preprocessor = joblib.load('marketing_campaign_model_preprocessor.pkl')
    logging.info("Successfully loaded ML models and preprocessor")
except Exception as e:
    logging.error(f"Error loading ML models: {e}")
    models = None
    preprocessor = None

# Request Model
class CampaignInput(BaseModel):
    campaign_type: str
    customer_segment: str
    location: str
    language: str
    duration_days: int
    month: int = None
    year: int = None
    conversion_weight: float = 1.0
    roi_weight: float = 1.0
    engagement_weight: float = 1.0

# Response Model
class CampaignRecommendation(BaseModel):
    channel_used: str
    target_audience: str
    predicted_conversion_rate: float
    predicted_roi: float
    predicted_engagement: float
    overall_score: float

# Historical Response Model
class HistoricalCampaign(CampaignInput):
    id: int
    created_at: datetime

# Initialize FastAPI App
app = FastAPI(title="Marketing Campaign Optimizer API")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Marketing Campaign Optimizer API!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running correctly"}

def generate_recommendations(campaign: CampaignInput):
    if models is None or preprocessor is None:
        logging.error("ML models not loaded")
        raise HTTPException(status_code=500, detail="ML models not available")

    channels = ['Google Ads', 'YouTube', 'Instagram', 'Website', 'Facebook', 'Email']
    audiences = ['Men 18-24', 'Men 25-34', 'Women 25-34', 'Women 35-44', 'All Ages']

    # Pre-calculate common values
    current_month = datetime.now().month
    current_year = datetime.now().year
    month = campaign.month if campaign.month else current_month
    year = campaign.year if campaign.year else current_year

    # Prepare all test cases at once using list comprehension
    test_cases = [{
        'Campaign_Type': campaign.campaign_type,
        'Target_Audience': audience,
        'Duration_Days': campaign.duration_days,
        'Channel_Used': channel,
        'Location': campaign.location,
        'Language': campaign.language,
        'Customer_Segment': campaign.customer_segment,
        'Month': month,
        'Year': year
    } for channel in channels for audience in audiences]

    # Convert to DataFrame in one operation
    test_df = pd.DataFrame(test_cases)

    try:
        results = []
        for _, row in test_df.iterrows():
            row_df = pd.DataFrame([row])
            
            # Predict all three metrics
            pred_conversion = float(models['conversion_rate'].predict(row_df)[0])
            pred_roi = float(models['roi'].predict(row_df)[0])
            pred_engagement = float(models['engagement'].predict(row_df)[0])

            overall_score = (
                pred_conversion * campaign.conversion_weight +
                pred_roi * campaign.roi_weight +
                pred_engagement * campaign.engagement_weight
            )

            results.append({
                'channel_used': row['Channel_Used'],
                'target_audience': row['Target_Audience'],
                'predicted_conversion_rate': pred_conversion,
                'predicted_roi': pred_roi,
                'predicted_engagement': pred_engagement,
                'overall_score': overall_score
            })

        # Sort once at the end
        results_sorted = sorted(results, key=lambda x: x['overall_score'], reverse=True)
        return results_sorted

    except Exception as e:
        logging.error(f"Model Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error in model prediction")

@app.post("/optimize", response_model=list[CampaignRecommendation])
async def optimize_campaign(
    campaign: CampaignInput,
    db: Session = Depends(get_db)
):
    """
    Generate marketing campaign recommendations based on user input.
    Returns top recommendations with predicted metrics.
    """
    try:
        # Save input to database
        db_campaign = DBCampaignInput(**campaign.dict())
        db.add(db_campaign)
        db.commit()
        db.refresh(db_campaign)
        
        logging.info(f"Saved campaign input to database with ID: {db_campaign.id}")
        
        # Generate recommendations in thread pool
        start_time = time.time()
        recommendations = await run_in_threadpool(generate_recommendations, campaign)
        logging.info(f"Recommendation generation took {time.time()-start_time:.2f} seconds")
        
        return recommendations[:10]

    except ValidationError as ve:
        db.rollback()
        logging.error(f"Validation Error: {str(ve)}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        db.rollback()
        logging.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/historical", response_model=List[HistoricalCampaign])
async def get_historical_campaigns(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """
    Retrieve historical campaign inputs from the database
    """
    campaigns = db.query(DBCampaignInput).offset(skip).limit(limit).all()
    return campaigns

# Handle OPTIONS requests explicitly
@app.options("/optimize")
async def options_optimize():
    return {"message": "OK"}

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {str(exc)}")
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )