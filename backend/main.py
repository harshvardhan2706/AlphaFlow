import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from typing import Dict, List
from data_loader import read_csv_to_dataframe
from model.ohlcv_model import OHLCV
from pydantic import BaseModel, Field
from backend.strategy import indicators as ind
from backend.strategy.executor import execute_strategy
from backend.strategy.logic_builder import evaluate_logic
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

class StrategyRequest(BaseModel):
    indicators: list = Field(..., description="List of indicators to apply, e.g. [{name: 'ema', params: {period: 20, price_col: 'close', out_col: 'ema_20'}}]")
    logic: dict = Field(..., description="Logic rules with 'entry', 'exit', and 'conditions' list")
    execution: dict = Field(..., description="Execution params: order_type, stop_loss, take_profit, etc.")

@app.post("/run-strategy")
async def run_strategy(req: StrategyRequest = Body(...)):
    global ohlcv_data
    if ohlcv_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Upload data first.")
    df = ohlcv_data.copy()
    # Apply indicators
    for ind_cfg in req.indicators:
        name = ind_cfg['name'].lower()
        params = ind_cfg.get('params', {})
        if name == 'ema':
            df = ind.add_ema(df, **params)
        elif name == 'rsi':
            df = ind.add_rsi(df, **params)
        elif name == 'macd':
            df = ind.add_macd(df, **params)
        # Add more indicators as needed
    # Evaluate logic
    conditions = req.logic['conditions']
    entry_logic = req.logic['entry']
    exit_logic = req.logic['exit']
    entry_signal = evaluate_logic(df, conditions, entry_logic)
    exit_signal = evaluate_logic(df, conditions, exit_logic)
    # Execution params
    order_type = req.execution.get('order_type', 'market')
    stop_loss = req.execution.get('stop_loss')
    take_profit = req.execution.get('take_profit')
    # Run strategy
    result = execute_strategy(
        df,
        entry_signal,
        exit_signal,
        order_type=order_type,
        initial_balance=req.execution.get('initial_balance', 10000.0),
        position_size=req.execution.get('position_size', 1.0)
    )
    # Calculate Sharpe ratio
    returns = pd.Series(result['trades']).apply(lambda t: t.get('pnl', 0))
    if len(returns) > 1 and returns.std() != 0:
        sharpe = (returns.mean() / returns.std()) * (252 ** 0.5)
    else:
        sharpe = 0.0
    return {
        'trades': result['trades'],
        'metrics': {
            'total_pnl': result['total_pnl'],
            'max_drawdown': result['max_drawdown'],
            'final_balance': result['final_balance'],
            'total_trades': result['total_trades'],
            'sharpe_ratio': sharpe
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)