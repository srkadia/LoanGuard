import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipelines.predict_piepline import PredictPipeline
from utils.config import ConfigLoader
from utils.logger import Logger

# Initialize Logger
logger = Logger(__name__).get_logger()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods like GET, POST, etc.
    allow_headers=["*"],  # Allows all headers
)

# Define the request model with proper data types and conversions
class LoanApplication(BaseModel):
    loan_amnt: float
    term: str
    int_rate: float
    installment: float
    sub_grade: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    purpose: str
    dti: float
    earliest_cr_line: str
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    initial_list_status: str
    application_type: str
    mort_acc: float
    pub_rec_bankruptcies: float
    address: str

@app.post("/predict")
async def predict(input_data: LoanApplication):
    try:
        # Convert Pydantic model to dictionary
        user_input = input_data.model_dump()
        logger.info(user_input)

        # Load config and run pipeline
        config_file_path = os.path.abspath("config.yaml")
        config = ConfigLoader.load_config(config_file_path)
        pipeline = PredictPipeline(user_input, config)
        prediction = pipeline.run_pipeline().tolist()
        logger.info("Prediction: " + str(prediction[0][0]))

        return {"prediction": prediction[0][0]}
    except Exception as e:
        logger.error(HTTPException(status_code=500, detail=str(e)))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Loan Default Prediction API is running"}
