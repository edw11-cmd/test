#!/usr/bin/env python3
"""
HyperCortex-AI Main Application Entry Point
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from hypercortex.core.config import get_settings
from hypercortex.api.main import app
from hypercortex.agents.orchestrator import get_orchestrator
from hypercortex.memory.vector_store import get_memory_manager
from hypercortex.tools.builtin_tools import register_builtin_tools
from hypercortex.monitoring.metrics import get_metrics_manager
import structlog
import uvicorn

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def initialize_system():
    """Initialize the HyperCortex-AI system"""
    
    logger.info("Initializing HyperCortex-AI system...")
    
    # Get settings
    settings = get_settings()
    
    # Initialize core components
    orchestrator = get_orchestrator()
    memory_manager = get_memory_manager()
    metrics_manager = get_metrics_manager()
    
    # Register built-in tools
    register_builtin_tools()
    
    # Store some initial memories for demonstration
    await memory_manager.store_memory(
        content="HyperCortex-AI is an advanced multi-agent AI framework that uses OpenAI and Opik for observability.",
        metadata={"type": "system_info", "importance": "high"},
        category="system"
    )
    
    await memory_manager.store_memory(
        content="The system includes PlannerAgent for strategic planning, ResearchAgent for information gathering, and CoderAgent for software development.",
        metadata={"type": "agent_info", "importance": "high"},
        category="system"
    )
    
    logger.info("HyperCortex-AI system initialized successfully")
    
    return {
        "orchestrator": orchestrator,
        "memory_manager": memory_manager,
        "metrics_manager": metrics_manager
    }


async def run_example_tasks():
    """Run some example tasks to demonstrate the system"""
    
    logger.info("Running example tasks...")
    
    orchestrator = get_orchestrator()
    
    # Example 1: Research task
    logger.info("Example 1: Research Task")
    research_result = await orchestrator.execute_task(
        task_description="Research the latest trends in AI agent frameworks and multi-agent systems",
        task_type="research"
    )
    
    logger.info(f"Research task completed with status: {research_result.status}")
    print(f"\nResearch Task Result:\n{research_result.result.get('summary', 'No summary available')}\n")
    
    # Example 2: Planning task
    logger.info("Example 2: Planning Task")
    planning_result = await orchestrator.execute_task(
        task_description="Create a plan for building a web application that integrates OpenAI and a vector database",
        task_type="planning"
    )
    
    logger.info(f"Planning task completed with status: {planning_result.status}")
    print(f"\nPlanning Task Result:\n{planning_result.result.get('summary', 'No summary available')}\n")
    
    # Example 3: Coding task
    logger.info("Example 3: Coding Task")
    coding_result = await orchestrator.execute_task(
        task_description="Write a Python function that calculates the Fibonacci sequence using dynamic programming",
        task_type="coding"
    )
    
    logger.info(f"Coding task completed with status: {coding_result.status}")
    print(f"\nCoding Task Result:\n{coding_result.result.get('summary', 'No summary available')}\n")
    
    # Example 4: Complex multi-agent task
    logger.info("Example 4: Complex Multi-Agent Task")
    complex_result = await orchestrator.execute_task(
        task_description="Design and implement a simple REST API for a todo application with user authentication",
        task_type="complex"
    )
    
    logger.info(f"Complex task completed with status: {complex_result.status}")
    print(f"\nComplex Task Result:\n{complex_result.result.get('summary', 'No summary available')}\n")
    
    logger.info("Example tasks completed")


def main():
    """Main application entry point"""
    
    settings = get_settings()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            HyperCortex-AI                                    ║
║                   Advanced Multi-Agent AI Framework                          ║
║                                                                              ║
║  🤖 Multi-Agent System with OpenAI & Opik Integration                       ║
║  🧠 Memory-Augmented with Vector Storage                                     ║
║  🔧 Autonomous Task Decomposition & Tool Usage                              ║
║  🔄 Self-Healing Planning with ReAct + Reflexion                            ║
║  📊 Real-time Observability & Monitoring                                    ║
║                                                                              ║
║  Version: {settings.app_version:<10} | Port: {settings.port:<10} | Debug: {str(settings.debug):<10}        ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Check if we should run examples or start the server
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        # Run examples
        async def run_examples():
            await initialize_system()
            await run_example_tasks()
        
        asyncio.run(run_examples())
    
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        # Run a simple test
        async def run_test():
            components = await initialize_system()
            
            # Test memory system
            memory_manager = components["memory_manager"]
            
            # Store a test memory
            memory_id = await memory_manager.store_memory(
                content="This is a test memory for HyperCortex-AI",
                metadata={"test": True},
                category="test"
            )
            
            print(f"Stored test memory with ID: {memory_id}")
            
            # Search for memories
            results = await memory_manager.recall_memories("test memory", k=3)
            print(f"Found {len(results)} relevant memories")
            
            for memory, score in results:
                print(f"  - {memory.content[:50]}... (score: {score:.2f})")
            
            # Test orchestrator
            orchestrator = components["orchestrator"]
            agent_status = await orchestrator.get_agent_status()
            
            print(f"\nAvailable agents: {list(agent_status.keys())}")
            
            print("\n✅ System test completed successfully!")
        
        asyncio.run(run_test())
    
    else:
        # Start the FastAPI server
        logger.info(f"Starting HyperCortex-AI server on {settings.host}:{settings.port}")
        
        # Initialize system before starting server
        async def startup():
            await initialize_system()
        
        # Run startup
        asyncio.run(startup())
        
        # Start the server
        uvicorn.run(
            "hypercortex.api.main:app",
            host=settings.host,
            port=settings.port,
            reload=False,  # Disable reload to avoid import issues
            log_level=settings.log_level.lower(),
            access_log=True
        )


if __name__ == "__main__":
    main()