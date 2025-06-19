import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, List
from data_loader import read_csv_to_dataframe
from model.ohlcv_model import OHLCV
import logging

app = FastAPI(title="AlphaFlow API")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for OHLCV data
ohlcv_data: pd.DataFrame | None = None

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/load-data")
async def load_data(file: UploadFile) -> Dict[str, str]:
    """Load OHLCV data from CSV file"""
    global ohlcv_data
    
    if not file.filename.endswith('.csv'):
        logger.error(f"File upload failed: {file.filename} is not a CSV.")
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Use the data_loader function to parse the CSV and handle timestamp
        # Await file.read() to get the content, then wrap in UploadFile-like object
        class AsyncUploadFile:
            def __init__(self, filename, content):
                self.filename = filename
                from io import BytesIO
                self.file = BytesIO(content)
        
        content = await file.read()
        sync_file = AsyncUploadFile(file.filename, content)
        ohlcv_data = read_csv_to_dataframe(sync_file)
        
        # Validate required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in ohlcv_data.columns.str.lower()]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Standardize column names to lowercase
        ohlcv_data.columns = ohlcv_data.columns.str.lower()
        
        logger.info(f"Data loaded successfully from {file.filename}.")
        return {"message": "Data loaded successfully"}
    
    except Exception as e:
        logger.exception(f"Error loading data: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get-data", response_model=Dict[str, List[OHLCV]])
async def get_data() -> Dict[str, List[OHLCV]]:
    """Retrieve loaded OHLCV data as a list of OHLCV models"""
    if ohlcv_data is None:
        logger.error("No data has been loaded when calling /get-data.")
        raise HTTPException(status_code=404, detail="No data has been loaded")
    
    try:
        df = ohlcv_data.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(str)
        data = [OHLCV(**row) for row in df.to_dict(orient='records')]
        logger.info(f"Returned {len(data)} OHLCV records.")
        return {"data": data}
    
    except Exception as e:
        logger.exception(f"Error returning data: {e}")
        raise HTTPException(status_code=500, detail="Failed to return data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)