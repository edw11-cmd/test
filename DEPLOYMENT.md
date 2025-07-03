# HyperCortex-AI Deployment Guide

## 🚀 Quick Deployment

### Local Development

1. **Clone and Setup**
```bash
git clone <repository-url>
cd hypercortex-ai
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_openai_api_key
# OPIK_API_KEY=your_opik_api_key (optional)
```

3. **Run the System**
```bash
# Test the system
python main.py test

# Start the API server
python main.py

# Access the web UI
open http://localhost:12000/ui
```

### Docker Deployment

1. **Using Docker Compose (Recommended)**
```bash
# Set your API keys in .env file
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

2. **Using Docker directly**
```bash
# Build the image
docker build -t hypercortex-ai .

# Run the container
docker run -d \
  --name hypercortex-ai \
  -p 12000:12000 \
  -e OPENAI_API_KEY=your_key \
  -e OPIK_API_KEY=your_key \
  hypercortex-ai

# View logs
docker logs -f hypercortex-ai
```

## 🌐 Production Deployment

### Cloud Platforms

#### 1. Fly.io Deployment

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
flyctl auth login

# Deploy the application
flyctl deploy

# Set environment variables
flyctl secrets set OPENAI_API_KEY=your_key
flyctl secrets set OPIK_API_KEY=your_key

# View logs
flyctl logs
```

#### 2. Render Deployment

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Set the following:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python main.py`
   - **Environment Variables**:
     - `OPENAI_API_KEY`: your_openai_api_key
     - `OPIK_API_KEY`: your_opik_api_key
     - `HOST`: 0.0.0.0
     - `PORT`: 10000

#### 3. Railway Deployment

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up

# Set environment variables
railway variables set OPENAI_API_KEY=your_key
railway variables set OPIK_API_KEY=your_key
```

#### 4. Heroku Deployment

```bash
# Install Heroku CLI
# Create Procfile
echo "web: python main.py" > Procfile

# Initialize git and Heroku
git init
heroku create your-app-name

# Set environment variables
heroku config:set OPENAI_API_KEY=your_key
heroku config:set OPIK_API_KEY=your_key
heroku config:set HOST=0.0.0.0
heroku config:set PORT=\$PORT

# Deploy
git add .
git commit -m "Initial deployment"
git push heroku main
```

### AWS/GCP/Azure Deployment

#### AWS ECS with Fargate

1. **Build and push Docker image to ECR**
```bash
# Create ECR repository
aws ecr create-repository --repository-name hypercortex-ai

# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and tag image
docker build -t hypercortex-ai .
docker tag hypercortex-ai:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/hypercortex-ai:latest

# Push image
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/hypercortex-ai:latest
```

2. **Create ECS Task Definition**
```json
{
  "family": "hypercortex-ai",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<account-id>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "hypercortex-ai",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/hypercortex-ai:latest",
      "portMappings": [
        {
          "containerPort": 12000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "HOST",
          "value": "0.0.0.0"
        },
        {
          "name": "PORT",
          "value": "12000"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:<account-id>:secret:openai-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/hypercortex-ai",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Google Cloud Run

```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/hypercortex-ai

# Deploy to Cloud Run
gcloud run deploy hypercortex-ai \
  --image gcr.io/PROJECT-ID/hypercortex-ai \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars HOST=0.0.0.0,PORT=8080 \
  --set-secrets OPENAI_API_KEY=openai-key:latest
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name hypercortex-rg --location eastus

# Create container instance
az container create \
  --resource-group hypercortex-rg \
  --name hypercortex-ai \
  --image your-registry/hypercortex-ai:latest \
  --cpu 1 \
  --memory 2 \
  --ports 12000 \
  --environment-variables HOST=0.0.0.0 PORT=12000 \
  --secure-environment-variables OPENAI_API_KEY=your_key
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `OPIK_API_KEY` | Opik API key | - | No |
| `OPIK_WORKSPACE` | Opik workspace name | hypercortex-ai | No |
| `HOST` | Server host | 0.0.0.0 | No |
| `PORT` | Server port | 12000 | No |
| `DEBUG` | Debug mode | True | No |
| `LOG_LEVEL` | Logging level | INFO | No |
| `OPENAI_MODEL` | Default OpenAI model | gpt-4-turbo-preview | No |
| `VECTOR_STORE_TYPE` | Vector store type | faiss | No |
| `MAX_MEMORY_SIZE` | Max memory entries | 10000 | No |
| `MEMORY_SIMILARITY_THRESHOLD` | Memory similarity threshold | 0.7 | No |
| `MAX_ITERATIONS` | Max agent iterations | 10 | No |
| `REFLECTION_ENABLED` | Enable agent reflection | True | No |

### Production Configuration

For production deployments, ensure:

1. **Security**
   - Set `DEBUG=False`
   - Use HTTPS (configure reverse proxy)
   - Implement authentication if needed
   - Secure API keys in secret management

2. **Performance**
   - Set appropriate CPU/memory limits
   - Configure Redis for memory storage
   - Enable connection pooling
   - Set up monitoring and alerting

3. **Reliability**
   - Configure health checks
   - Set up auto-scaling
   - Implement graceful shutdown
   - Configure backup strategies

### Redis Configuration (Optional)

For production, use Redis for memory storage:

```bash
# Set Redis URL
export REDIS_URL=redis://localhost:6379/0

# Or for Redis Cloud
export REDIS_URL=redis://username:password@host:port/0
```

## 📊 Monitoring

### Health Checks

- **Health endpoint**: `GET /health`
- **Metrics endpoint**: `GET /metrics`
- **System info**: `GET /`

### Logging

The system uses structured logging with the following levels:
- `DEBUG`: Detailed debugging information
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error messages

### Metrics

Available metrics include:
- System uptime
- Request count and response times
- Agent performance metrics
- Memory usage statistics
- Tool execution metrics

## 🔒 Security

### API Security

1. **Rate Limiting**: Implement rate limiting for production
2. **Authentication**: Add JWT or API key authentication
3. **CORS**: Configure CORS appropriately for your domain
4. **Input Validation**: All inputs are validated using Pydantic

### Infrastructure Security

1. **Network Security**: Use VPCs and security groups
2. **Secret Management**: Use cloud secret managers
3. **Container Security**: Scan images for vulnerabilities
4. **Access Control**: Implement least privilege access

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   ```bash
   # Verify API key is set
   echo $OPENAI_API_KEY
   
   # Test API key
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   curl http://localhost:12000/metrics
   
   # Clear memory if needed
   # (implement clear endpoint if required)
   ```

3. **Port Conflicts**
   ```bash
   # Check if port is in use
   lsof -i :12000
   
   # Use different port
   export PORT=8080
   python main.py
   ```

4. **Docker Issues**
   ```bash
   # Check container logs
   docker logs hypercortex-ai
   
   # Check container status
   docker ps -a
   
   # Restart container
   docker restart hypercortex-ai
   ```

### Performance Optimization

1. **Memory Optimization**
   - Reduce `MAX_MEMORY_SIZE` if memory usage is high
   - Use Redis for distributed memory storage
   - Implement memory cleanup strategies

2. **Response Time Optimization**
   - Use faster OpenAI models for simple tasks
   - Implement caching for frequent requests
   - Optimize agent iteration limits

3. **Scaling**
   - Use horizontal scaling with load balancers
   - Implement connection pooling
   - Use async processing for long-running tasks

## 📞 Support

For deployment issues:
1. Check the logs for error messages
2. Verify environment variables are set correctly
3. Test API endpoints individually
4. Check system resources (CPU, memory, disk)
5. Review network connectivity

For additional support, please refer to the main README.md or create an issue in the repository.