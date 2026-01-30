# Deployment Guide

This guide covers different deployment options for LLM Guardrail Studio.

## Local Development

### Prerequisites

- Python 3.8+
- Git
- 4GB+ RAM (8GB+ recommended for local models)
- GPU (optional, for faster inference)

### Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/Ask-812/LLM-Guardrail-Studio.git
   cd LLM-Guardrail-Studio
   pip install -r requirements.txt
   ```

2. **Run dashboard**
   ```bash
   streamlit run app.py
   ```

3. **Access at** `http://localhost:8501`

## Docker Deployment

### Basic Docker Setup

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
   ```

2. **Build and run**
   ```bash
   docker build -t guardrail-studio .
   docker run -p 8501:8501 guardrail-studio
   ```

### Docker Compose

```yaml
version: '3.8'

services:
  guardrail-studio:
    build: .
    ports:
      - "8501:8501"
    environment:
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

Run with: `docker-compose up -d`

## Cloud Deployment

### Streamlit Cloud

1. **Push to GitHub**
2. **Connect to Streamlit Cloud**
3. **Deploy from repository**
4. **Configure secrets** in Streamlit Cloud dashboard

### Heroku

1. **Create Procfile**
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```

2. **Deploy**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### AWS EC2

1. **Launch EC2 instance** (t3.medium or larger)

2. **Install dependencies**
   ```bash
   sudo apt update
   sudo apt install python3-pip git
   git clone https://github.com/Ask-812/LLM-Guardrail-Studio.git
   cd LLM-Guardrail-Studio
   pip3 install -r requirements.txt
   ```

3. **Run with systemd**
   ```bash
   # Create service file
   sudo nano /etc/systemd/system/guardrail-studio.service
   ```
   
   ```ini
   [Unit]
   Description=LLM Guardrail Studio
   After=network.target
   
   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/LLM-Guardrail-Studio
   ExecStart=/usr/local/bin/streamlit run app.py --server.address 0.0.0.0
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```
   
   ```bash
   sudo systemctl enable guardrail-studio
   sudo systemctl start guardrail-studio
   ```

4. **Configure nginx** (optional)
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Google Cloud Run

1. **Create Dockerfile** (see Docker section)

2. **Build and deploy**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT-ID/guardrail-studio
   gcloud run deploy --image gcr.io/PROJECT-ID/guardrail-studio --platform managed
   ```

## API Deployment

### FastAPI Wrapper

Create `api.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from guardrails import GuardrailPipeline

app = FastAPI(title="LLM Guardrail API")
pipeline = GuardrailPipeline()

class EvaluationRequest(BaseModel):
    prompt: str
    response: str

class EvaluationResponse(BaseModel):
    passed: bool
    scores: dict
    flags: list

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    result = pipeline.evaluate(request.prompt, request.response)
    return EvaluationResponse(
        passed=result.passed,
        scores=result.scores,
        flags=result.flags
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Deploy with:
```bash
pip install fastapi uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Production Considerations

### Performance Optimization

1. **Model Caching**
   ```python
   # Use model caching for better performance
   @st.cache_resource
   def load_pipeline():
       return GuardrailPipeline()
   ```

2. **Batch Processing**
   - Process multiple evaluations together
   - Use async processing for better throughput

3. **GPU Acceleration**
   - Use CUDA-enabled containers
   - Configure GPU memory management

### Security

1. **Input Validation**
   - Sanitize user inputs
   - Implement rate limiting
   - Add authentication if needed

2. **Environment Variables**
   ```bash
   export GUARDRAIL_MODEL_PATH=/path/to/models
   export GUARDRAIL_LOG_LEVEL=INFO
   ```

3. **HTTPS Configuration**
   - Use SSL certificates
   - Configure secure headers

### Monitoring

1. **Logging**
   ```python
   import logging
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   )
   ```

2. **Metrics Collection**
   - Track evaluation counts
   - Monitor response times
   - Log error rates

3. **Health Checks**
   ```python
   @app.get("/health")
   async def health_check():
       return {"status": "healthy"}
   ```

### Scaling

1. **Horizontal Scaling**
   - Use load balancers
   - Deploy multiple instances
   - Implement session management

2. **Vertical Scaling**
   - Increase memory/CPU
   - Use larger instance types
   - Optimize model loading

## Environment Variables

Common environment variables:

```bash
# Model configuration
GUARDRAIL_MODEL_NAME=mistralai/Mistral-7B-v0.1
GUARDRAIL_DEVICE=cuda

# Thresholds
GUARDRAIL_TOXICITY_THRESHOLD=0.7
GUARDRAIL_ALIGNMENT_THRESHOLD=0.5

# Performance
GUARDRAIL_BATCH_SIZE=32
GUARDRAIL_MAX_LENGTH=512

# Logging
GUARDRAIL_LOG_LEVEL=INFO
GUARDRAIL_LOG_FILE=/var/log/guardrail.log
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce model size
   - Use CPU instead of GPU
   - Implement model unloading

2. **Slow Performance**
   - Enable model caching
   - Use smaller models
   - Implement batch processing

3. **Import Errors**
   - Check Python version
   - Verify all dependencies installed
   - Use virtual environment

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

Run with verbose output:
```bash
streamlit run app.py --logger.level debug
```