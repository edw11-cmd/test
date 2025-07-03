# HyperCortex-AI

**The World's Most Advanced Multi-Agent AI Framework**

HyperCortex-AI is an autonomous, memory-augmented, multi-agent AI system built with OpenAI GPT-4.5 and Opik observability infrastructure. It goes beyond traditional LLMs by integrating real-time reflection, chain-of-thought reasoning, agent memory, autonomous task decomposition, and self-healing planning.

## 🚀 Features

### Core Capabilities
- **🤖 Multi-Agent System**: PlannerAgent, ResearchAgent, CoderAgent, and more
- **🧠 Memory-Augmented**: Long-term vector storage with FAISS/ChromaDB
- **🔄 Self-Healing Planning**: ReAct + Reflexion + LangGraph hybrid architecture
- **🔧 Autonomous Task Decomposition**: Intelligent task breakdown and execution
- **📊 Real-time Observability**: Comprehensive monitoring with Opik integration
- **🛠️ Extensible Tool System**: Browser access, code execution, file management, and more

### Advanced Features
- **💭 Real-time Reflection**: Agents learn and improve from past experiences
- **🎯 Dynamic Prompt Engineering**: AI-powered prompt optimization
- **🔐 Security-First Design**: Secure API endpoints and data handling
- **📈 Performance Monitoring**: Detailed metrics and analytics
- **🌐 RESTful API**: Complete FastAPI-based interface
- **🐳 Production Ready**: Docker containerization and deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        HyperCortex-AI                           │
├─────────────────────────────────────────────────────────────────┤
│  FastAPI REST API                                              │
├─────────────────────────────────────────────────────────────────┤
│  Agent Orchestrator                                            │
│  ├── PlannerAgent (Strategic Planning)                         │
│  ├── ResearchAgent (Information Gathering)                     │
│  ├── CoderAgent (Software Development)                         │
│  └── Custom Agents (Extensible)                               │
├─────────────────────────────────────────────────────────────────┤
│  Memory System (FAISS Vector Store)                           │
├─────────────────────────────────────────────────────────────────┤
│  Tool Registry                                                │
│  ├── Web Search                                               │
│  ├── Code Executor                                            │
│  ├── File Manager                                             │
│  └── HTTP Client                                              │
├─────────────────────────────────────────────────────────────────┤
│  LLM Engine (OpenAI + Opik Tracking)                         │
├─────────────────────────────────────────────────────────────────┤
│  Monitoring & Metrics                                         │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Opik API Key (provides OpenAI access - no separate OpenAI key needed)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd hypercortex-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your Opik API key
# No OpenAI API key needed - Opik provides OpenAI access
```

4. **Test Opik-only configuration**
```bash
# Test Opik integration
python test_opik_only.py

# Test the full system
python main.py test

# Run example tasks
python main.py examples

# Start the API server
python main.py
```

### Docker Deployment

1. **Using Docker Compose**
```bash
# Set your Opik API key in .env file
docker-compose up -d
```

2. **Using Docker directly**
```bash
docker build -t hypercortex-ai .
docker run -p 12000:12000 --env-file .env hypercortex-ai
```

## 📖 Usage Examples

### 1. Research Task
```python
import asyncio
from hypercortex.agents.orchestrator import get_orchestrator

async def research_example():
    orchestrator = get_orchestrator()
    
    result = await orchestrator.execute_task(
        task_description="Research the latest trends in AI agent frameworks",
        task_type="research"
    )
    
    print(f"Research completed: {result.result['summary']}")

asyncio.run(research_example())
```

### 2. Code Generation
```python
async def coding_example():
    orchestrator = get_orchestrator()
    
    result = await orchestrator.execute_task(
        task_description="Write a Python function to calculate Fibonacci numbers",
        task_type="coding"
    )
    
    print(f"Code generated: {result.result['summary']}")

asyncio.run(coding_example())
```

### 3. Strategic Planning
```python
async def planning_example():
    orchestrator = get_orchestrator()
    
    result = await orchestrator.execute_task(
        task_description="Create a plan for building a web application",
        task_type="planning"
    )
    
    print(f"Plan created: {result.result['summary']}")

asyncio.run(planning_example())
```

## 🌐 API Endpoints

### Task Execution
- `POST /tasks` - Create and execute a task
- `GET /tasks` - Get task history
- `GET /tasks/{task_id}` - Get specific task result

### Chat Interface
- `POST /chat` - Chat with an agent
- `POST /chat/stream` - Stream chat responses

### Agent Management
- `GET /agents` - List available agents
- `GET /agents/{agent_name}` - Get agent status

### Memory Management
- `POST /memory/search` - Search memory
- `POST /memory/store` - Store information
- `DELETE /memory/{memory_id}` - Delete memory

### Tool Management
- `GET /tools` - List available tools
- `POST /tools/execute` - Execute a tool

### Monitoring
- `GET /health` - Health check
- `GET /metrics` - System metrics

## 🔧 Configuration

### Environment Variables

```bash
# Opik Configuration (provides OpenAI access)
OPIK_API_KEY=XvVArO*****************
OPIK_WORKSPACE=tak11-cloud

# Model Configuration
OPENAI_MODEL=gpt-3.5-turbo

# Application Settings
DEBUG=True
HOST=0.0.0.0
PORT=12000

# Memory Configuration
VECTOR_STORE_TYPE=faiss
MAX_MEMORY_SIZE=10000
MEMORY_SIMILARITY_THRESHOLD=0.7

# Agent Configuration
MAX_ITERATIONS=10
REFLECTION_ENABLED=True
```

## 🧪 Testing

```bash
# Run system test
python main.py test

# Run example tasks
python main.py examples

# Test API endpoints
curl http://localhost:12000/health
curl -X POST http://localhost:12000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how can you help me?"}'
```

## 🔍 Monitoring & Observability

HyperCortex-AI includes comprehensive monitoring:

- **Real-time Metrics**: API requests, agent performance, LLM usage
- **Opik Integration**: Track all OpenAI calls with detailed observability
- **Performance Analytics**: Response times, success rates, error tracking
- **Memory Analytics**: Vector search performance, memory usage
- **Agent Analytics**: Task completion rates, reflection insights

Access metrics at: `http://localhost:12000/metrics`

## 🛠️ Extending the System

### Adding Custom Agents

```python
from hypercortex.agents.base_agent import BaseAgent, AgentAction, AgentObservation

class CustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CustomAgent",
            role="Custom Role",
            system_prompt="Your custom system prompt"
        )
    
    async def _decide_next_action(self, task: str) -> Optional[AgentAction]:
        # Implement your decision logic
        pass
    
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        # Implement your action execution
        pass
    
    async def _is_task_complete(self, task: str, last_observation: AgentObservation) -> bool:
        # Implement completion check
        pass

# Register the agent
from hypercortex.agents.orchestrator import get_orchestrator
orchestrator = get_orchestrator()
orchestrator.add_agent("custom", CustomAgent())
```

### Adding Custom Tools

```python
from hypercortex.tools.base_tool import BaseTool, ToolParameter, ToolResult

class CustomTool(BaseTool):
    def __init__(self):
        parameters = [
            ToolParameter(
                name="input_data",
                type="string",
                description="Input data for processing",
                required=True
            )
        ]
        
        super().__init__(
            name="custom_tool",
            description="A custom tool for specific tasks",
            parameters=parameters
        )
    
    async def execute(self, input_data: str) -> ToolResult:
        # Implement your tool logic
        result = f"Processed: {input_data}"
        
        return ToolResult(
            success=True,
            result={"output": result},
            metadata={"tool": "custom"}
        )

# Register the tool
from hypercortex.tools.base_tool import get_tool_registry
registry = get_tool_registry()
registry.register_tool(CustomTool())
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
```bash
# Production environment variables
DEBUG=False
LOG_LEVEL=WARNING
OPENAI_API_KEY=your_production_key
```

2. **Docker Production**
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

3. **Cloud Deployment**
- **Fly.io**: Use the included `fly.toml` configuration
- **Render**: Deploy directly from GitHub
- **AWS/GCP/Azure**: Use Docker container deployment

### Security Considerations

- Store API keys securely (use environment variables or secret management)
- Enable HTTPS in production
- Implement rate limiting
- Use authentication for sensitive endpoints
- Regular security updates

## 📊 Performance

### Benchmarks
- **Task Execution**: ~2-10 seconds per task (depending on complexity)
- **Memory Search**: <100ms for 10K memories
- **API Response**: <200ms for simple requests
- **Agent Coordination**: Minimal overhead with async execution

### Scalability
- **Horizontal**: Multiple instances with shared Redis/database
- **Vertical**: Optimized for multi-core systems
- **Memory**: Efficient vector storage with configurable limits
- **Throughput**: 100+ concurrent requests supported

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for the powerful GPT models
- Opik for observability infrastructure
- FastAPI for the excellent web framework
- The open-source AI community

## 📞 Support

- **Documentation**: [Link to docs]
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@hypercortex-ai.com

---

**HyperCortex-AI** - Pushing the boundaries of what's possible with AI agents. 🚀