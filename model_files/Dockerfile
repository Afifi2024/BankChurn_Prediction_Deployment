FROM python:3.9-slim

WORKDIR /app

# (Optional) Install system dependencies required for plotting or Keras
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all necessary files from model_files to the container workdir
COPY bank_churn_best_model.h5 ./
COPY label_encoders.joblib ./
COPY model_columns.joblib ./
COPY scaler.joblib ./
COPY smote.joblib ./
COPY app.py ./

EXPOSE 7860

CMD ["python", "app.py"]
