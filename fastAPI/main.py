from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
from main_dl_car_damage_detection import CarClassifierResNet, preprocess_image, predict_damage, load_model, get_class_labels

# Define the FastAPI app
app = FastAPI()

# Load model and class labels once during startup
model = load_model()
class_labels = get_class_labels()

@app.get("/")
async def index():
    print("Request received")
    return JSONResponse(content={"message": "Welcome to the Car Damage Detection API!"})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Log the file name for debugging
        print(f"Received file: {file.filename}")
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Log the file size
        print(f"Image size: {len(image_bytes)} bytes")

        predicted_label = predict_damage(image, model, class_labels)
        return JSONResponse(content={"prediction": predicted_label})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


