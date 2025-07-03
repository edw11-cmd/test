"""
FastAPI Main Application for HyperCortex-AI
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog

from ..core.config import get_settings
from ..agents.orchestrator import get_orchestrator
from ..memory.vector_store import get_memory_manager
from ..tools.base_tool import get_tool_registry
from ..tools.builtin_tools import register_builtin_tools
from ..monitoring.metrics import get_metrics_manager

logger = structlog.get_logger(__name__)
settings = get_settings()


# Pydantic models for API
class TaskRequest(BaseModel):
    description: str
    task_type: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    required_agents: Optional[List[str]] = None


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskResult(BaseModel):
    task_id: str
    task_description: str
    status: str
    result: Dict[str, Any]
    agents_used: List[str]
    execution_time: float
    timestamp: str


class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    agent: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    metadata: Dict[str, Any]


class MemoryQuery(BaseModel):
    query: str
    k: int = 5
    category: Optional[str] = None
    threshold: Optional[float] = None


class MemoryEntry(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    category: str = "general"


class ToolExecution(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    
    # Startup
    logger.info("Starting HyperCortex-AI application")
    
    # Register built-in tools
    register_builtin_tools()
    
    # Initialize metrics
    metrics_manager = get_metrics_manager()
    
    logger.info("HyperCortex-AI application started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down HyperCortex-AI application")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced Multi-Agent AI Framework with OpenAI and Opik Integration",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency injection
def get_orchestrator_dep():
    return get_orchestrator()


def get_memory_manager_dep():
    return get_memory_manager()


def get_tool_registry_dep():
    return get_tool_registry()


# API Routes

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Advanced Multi-Agent AI Framework",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    orchestrator = get_orchestrator()
    memory_manager = get_memory_manager()
    tool_registry = get_tool_registry()
    
    return {
        "status": "healthy",
        "agents": len(orchestrator.agents),
        "tools": len(tool_registry.tools),
        "memory_initialized": memory_manager is not None,
        "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
    }


# Task Execution Endpoints

@app.post("/tasks", response_model=TaskResponse)
async def create_task(
    task_request: TaskRequest,
    background_tasks: BackgroundTasks,
    orchestrator = Depends(get_orchestrator_dep)
):
    """Create and execute a new task"""
    
    try:
        # Start task execution in background
        task_result = await orchestrator.execute_task(
            task_description=task_request.description,
            task_type=task_request.task_type,
            context=task_request.context,
            required_agents=task_request.required_agents
        )
        
        return TaskResponse(
            task_id=task_result.task_id,
            status=task_result.status,
            message=f"Task {task_result.status}"
        )
        
    except Exception as e:
        logger.error(f"Error creating task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks", response_model=List[TaskResult])
async def get_tasks(
    limit: int = 10,
    orchestrator = Depends(get_orchestrator_dep)
):
    """Get task execution history"""
    
    try:
        task_history = await orchestrator.get_task_history(limit)
        
        return [
            TaskResult(
                task_id=task.task_id,
                task_description=task.task_description,
                status=task.status,
                result=task.result,
                agents_used=task.agents_used,
                execution_time=task.execution_time,
                timestamp=task.timestamp.isoformat()
            )
            for task in task_history
        ]
        
    except Exception as e:
        logger.error(f"Error getting tasks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks/{task_id}", response_model=TaskResult)
async def get_task(
    task_id: str,
    orchestrator = Depends(get_orchestrator_dep)
):
    """Get specific task result"""
    
    try:
        task_history = await orchestrator.get_task_history(100)  # Get more history
        
        task = next((t for t in task_history if t.task_id == task_id), None)
        
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return TaskResult(
            task_id=task.task_id,
            task_description=task.task_description,
            status=task.status,
            result=task.result,
            agents_used=task.agents_used,
            execution_time=task.execution_time,
            timestamp=task.timestamp.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task {task_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Chat Endpoints

@app.post("/chat", response_model=ChatResponse)
async def chat(
    chat_message: ChatMessage,
    orchestrator = Depends(get_orchestrator_dep)
):
    """Chat with an agent"""
    
    try:
        # Determine which agent to use
        agent_name = chat_message.agent or "researcher"  # Default to researcher
        
        if agent_name not in orchestrator.agents:
            raise HTTPException(status_code=400, detail=f"Agent '{agent_name}' not available")
        
        # Execute chat as a task
        task_result = await orchestrator.execute_task(
            task_description=chat_message.message,
            context=chat_message.context,
            required_agents=[agent_name]
        )
        
        # Extract response from task result
        response_content = task_result.result.get("summary", "No response generated")
        
        return ChatResponse(
            response=response_content,
            agent_used=agent_name,
            metadata={
                "task_id": task_result.task_id,
                "execution_time": task_result.execution_time,
                "status": task_result.status
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(
    chat_message: ChatMessage,
    orchestrator = Depends(get_orchestrator_dep)
):
    """Stream chat response"""
    
    async def generate_response():
        try:
            # This is a simplified streaming implementation
            # In a real implementation, you'd stream the agent's thinking process
            
            agent_name = chat_message.agent or "researcher"
            
            yield f"data: {{'type': 'start', 'agent': '{agent_name}'}}\n\n"
            
            # Execute task
            task_result = await orchestrator.execute_task(
                task_description=chat_message.message,
                context=chat_message.context,
                required_agents=[agent_name]
            )
            
            # Stream the response
            response = task_result.result.get("summary", "No response generated")
            
            # Simulate streaming by sending chunks
            words = response.split()
            for i, word in enumerate(words):
                yield f"data: {{'type': 'token', 'content': '{word} '}}\n\n"
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
            yield f"data: {{'type': 'end', 'task_id': '{task_result.task_id}'}}\n\n"
            
        except Exception as e:
            yield f"data: {{'type': 'error', 'error': '{str(e)}'}}\n\n"
    
    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# Agent Management Endpoints

@app.get("/agents")
async def get_agents(orchestrator = Depends(get_orchestrator_dep)):
    """Get available agents and their status"""
    
    try:
        agent_status = await orchestrator.get_agent_status()
        return agent_status
        
    except Exception as e:
        logger.error(f"Error getting agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/agents/{agent_name}")
async def get_agent(
    agent_name: str,
    orchestrator = Depends(get_orchestrator_dep)
):
    """Get specific agent status"""
    
    try:
        agent_status = await orchestrator.get_agent_status()
        
        if agent_name not in agent_status:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return agent_status[agent_name]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Memory Management Endpoints

@app.post("/memory/search")
async def search_memory(
    query: MemoryQuery,
    memory_manager = Depends(get_memory_manager_dep)
):
    """Search memory for relevant information"""
    
    try:
        results = await memory_manager.recall_memories(
            query=query.query,
            k=query.k,
            category=query.category,
            threshold=query.threshold
        )
        
        return {
            "query": query.query,
            "results": [
                {
                    "content": memory.content,
                    "metadata": memory.metadata,
                    "similarity_score": score,
                    "timestamp": memory.timestamp.isoformat()
                }
                for memory, score in results
            ],
            "count": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/store")
async def store_memory(
    memory_entry: MemoryEntry,
    memory_manager = Depends(get_memory_manager_dep)
):
    """Store information in memory"""
    
    try:
        memory_id = await memory_manager.store_memory(
            content=memory_entry.content,
            metadata=memory_entry.metadata,
            category=memory_entry.category
        )
        
        return {
            "memory_id": memory_id,
            "status": "stored"
        }
        
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str,
    memory_manager = Depends(get_memory_manager_dep)
):
    """Delete a memory entry"""
    
    try:
        success = await memory_manager.delete_memory(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"status": "deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Tool Management Endpoints

@app.get("/tools")
async def get_tools(tool_registry = Depends(get_tool_registry_dep)):
    """Get available tools"""
    
    try:
        return {
            "tools": tool_registry.get_tool_schemas(),
            "count": len(tool_registry.tools)
        }
        
    except Exception as e:
        logger.error(f"Error getting tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/{tool_name}")
async def get_tool(
    tool_name: str,
    tool_registry = Depends(get_tool_registry_dep)
):
    """Get specific tool information"""
    
    try:
        tool = tool_registry.get_tool(tool_name)
        
        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")
        
        return {
            "schema": tool.get_schema(),
            "stats": tool.get_stats()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tool {tool_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tools/execute")
async def execute_tool(
    tool_execution: ToolExecution,
    tool_registry = Depends(get_tool_registry_dep)
):
    """Execute a tool"""
    
    try:
        result = await tool_registry.execute_tool(
            tool_name=tool_execution.tool_name,
            **tool_execution.parameters
        )
        
        return {
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Error executing tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics and Monitoring Endpoints

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    
    try:
        metrics_manager = get_metrics_manager()
        return await metrics_manager.get_metrics()
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "hypercortex.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )