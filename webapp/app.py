from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pipelines.predict_piepline import PredictPipeline
from utils.config import ConfigLoader

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
        print(user_input)

        # Load config and run pipeline
        config = ConfigLoader.load_config("config.yaml")
        pipeline = PredictPipeline(user_input, config)
        prediction = pipeline.run_pipeline().tolist()

        return {"prediction": prediction[0][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Loan Default Prediction API is running"}
