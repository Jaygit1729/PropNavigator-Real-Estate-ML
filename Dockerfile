# Serving image for the PropNavigator price API.
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime; the slim base image doesn't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (they change rarely) so a code edit doesn't reinstall everything.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs at runtime.
COPY api/ ./api/
COPY artifacts/best_model.joblib ./artifacts/best_model.joblib
COPY data/price_prediction/ ./data/price_prediction/

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Start with a lightweight Python container
#            │
#            ▼
# Install required system library (libgomp1 for LightGBM)
#            │
#               ▼
# Set /app as the working folder
#            │
#            ▼
# Copy requirements file
#            │
#            ▼
# Install Python packages
#            │
#            ▼
# Copy API code
#            │
#            ▼
# Copy trained ML model
#            │
#            ▼
# Copy supporting data
#            │
#            ▼
# Tell Docker the app uses port 8000
#            │
#            ▼
# Start the FastAPI server with Uvicorn