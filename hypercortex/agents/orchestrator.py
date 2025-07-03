"""
Agent Orchestrator - Coordinates multi-agent workflows
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Type
from datetime import datetime
import structlog

from .base_agent import BaseAgent, AgentState
from .planner_agent import PlannerAgent
from .research_agent import ResearchAgent
from .coder_agent import CoderAgent
from ..memory.vector_store import get_memory_manager
from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class TaskResult:
    """Represents the result of a task execution"""
    
    def __init__(
        self,
        task_id: str,
        task_description: str,
        status: str,
        result: Dict[str, Any],
        agents_used: List[str],
        execution_time: float,
        metadata: Dict[str, Any] = None
    ):
        self.task_id = task_id
        self.task_description = task_description
        self.status = status
        self.result = result
        self.agents_used = agents_used
        self.execution_time = execution_time
        self.metadata = metadata or {}
        self.timestamp = datetime.utcnow()


class AgentOrchestrator:
    """Orchestrates multi-agent workflows and task execution"""
    
    def __init__(self):
        self.settings = settings
        self.memory_manager = get_memory_manager()
        
        # Initialize available agents
        self.agents: Dict[str, BaseAgent] = {
            "planner": PlannerAgent(),
            "researcher": ResearchAgent(),
            "coder": CoderAgent()
        }
        
        # Task execution history
        self.task_history: List[TaskResult] = []
        
        logger.info("Agent Orchestrator initialized with agents: " + ", ".join(self.agents.keys()))
    
    async def execute_task(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        context: Dict[str, Any] = None,
        required_agents: Optional[List[str]] = None
    ) -> TaskResult:
        """Execute a task using appropriate agents"""
        
        task_id = f"task_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.utcnow()
        
        logger.info(f"Starting task execution: {task_id}")
        
        try:
            # Determine execution strategy
            execution_plan = await self._create_execution_plan(
                task_description, task_type, required_agents
            )
            
            # Execute the plan
            result = await self._execute_plan(task_description, execution_plan, context)
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create task result
            task_result = TaskResult(
                task_id=task_id,
                task_description=task_description,
                status="completed" if result.get("success", False) else "failed",
                result=result,
                agents_used=execution_plan.get("agents", []),
                execution_time=execution_time,
                metadata={
                    "task_type": task_type,
                    "execution_plan": execution_plan,
                    "context": context
                }
            )
            
            # Store in history and memory
            self.task_history.append(task_result)
            await self._store_task_result(task_result)
            
            logger.info(f"Task {task_id} completed with status: {task_result.status}")
            return task_result
            
        except Exception as e:
            logger.error(f"Error executing task {task_id}: {str(e)}")
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            task_result = TaskResult(
                task_id=task_id,
                task_description=task_description,
                status="error",
                result={"error": str(e), "success": False},
                agents_used=[],
                execution_time=execution_time,
                metadata={"error": str(e)}
            )
            
            self.task_history.append(task_result)
            return task_result
    
    async def _create_execution_plan(
        self,
        task_description: str,
        task_type: Optional[str] = None,
        required_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create an execution plan for the task"""
        
        # If specific agents are required, use them
        if required_agents:
            return {
                "strategy": "specified_agents",
                "agents": required_agents,
                "execution_order": required_agents,
                "parallel_execution": False
            }
        
        # Determine task type if not specified
        if not task_type:
            task_type = await self._classify_task(task_description)
        
        # Create execution plan based on task type
        if task_type == "research":
            return {
                "strategy": "research_focused",
                "agents": ["researcher"],
                "execution_order": ["researcher"],
                "parallel_execution": False
            }
        
        elif task_type == "coding":
            return {
                "strategy": "coding_focused",
                "agents": ["coder"],
                "execution_order": ["coder"],
                "parallel_execution": False
            }
        
        elif task_type == "planning":
            return {
                "strategy": "planning_focused",
                "agents": ["planner"],
                "execution_order": ["planner"],
                "parallel_execution": False
            }
        
        elif task_type == "complex":
            return {
                "strategy": "multi_agent_sequential",
                "agents": ["planner", "researcher", "coder"],
                "execution_order": ["planner", "researcher", "coder"],
                "parallel_execution": False
            }
        
        else:  # Default strategy
            return {
                "strategy": "adaptive",
                "agents": ["planner", "researcher"],
                "execution_order": ["planner", "researcher"],
                "parallel_execution": False
            }
    
    async def _classify_task(self, task_description: str) -> str:
        """Classify the task type based on description"""
        
        task_lower = task_description.lower()
        
        # Coding keywords
        coding_keywords = [
            "code", "program", "implement", "develop", "build", "create app",
            "function", "class", "api", "database", "algorithm", "debug",
            "fix bug", "refactor", "optimize"
        ]
        
        # Research keywords
        research_keywords = [
            "research", "analyze", "study", "investigate", "explore",
            "find information", "compare", "evaluate", "assess",
            "market analysis", "competitive analysis", "trends"
        ]
        
        # Planning keywords
        planning_keywords = [
            "plan", "strategy", "roadmap", "design", "architecture",
            "approach", "methodology", "framework", "structure"
        ]
        
        # Complex task keywords
        complex_keywords = [
            "build system", "create platform", "develop solution",
            "end-to-end", "comprehensive", "full stack", "complete"
        ]
        
        # Check for complex tasks first
        if any(keyword in task_lower for keyword in complex_keywords):
            return "complex"
        
        # Check specific task types
        if any(keyword in task_lower for keyword in coding_keywords):
            return "coding"
        
        if any(keyword in task_lower for keyword in research_keywords):
            return "research"
        
        if any(keyword in task_lower for keyword in planning_keywords):
            return "planning"
        
        # Default to adaptive approach
        return "adaptive"
    
    async def _execute_plan(
        self,
        task_description: str,
        execution_plan: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute the planned workflow"""
        
        strategy = execution_plan.get("strategy", "sequential")
        agents_to_use = execution_plan.get("agents", [])
        execution_order = execution_plan.get("execution_order", agents_to_use)
        parallel_execution = execution_plan.get("parallel_execution", False)
        
        results = {}
        agent_outputs = {}
        
        try:
            if parallel_execution:
                # Execute agents in parallel
                tasks = []
                for agent_name in agents_to_use:
                    if agent_name in self.agents:
                        agent = self.agents[agent_name]
                        task_context = self._build_agent_context(context, agent_outputs)
                        tasks.append(agent.execute_task(task_description, task_context))
                
                # Wait for all agents to complete
                agent_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(agent_results):
                    agent_name = agents_to_use[i]
                    if isinstance(result, Exception):
                        results[agent_name] = {"error": str(result), "success": False}
                    else:
                        results[agent_name] = result
                        agent_outputs[agent_name] = result
            
            else:
                # Execute agents sequentially
                for agent_name in execution_order:
                    if agent_name not in self.agents:
                        logger.warning(f"Agent {agent_name} not available")
                        continue
                    
                    agent = self.agents[agent_name]
                    
                    # Build context including previous agent outputs
                    task_context = self._build_agent_context(context, agent_outputs)
                    
                    # Execute agent task
                    logger.info(f"Executing agent: {agent_name}")
                    agent_result = await agent.execute_task(task_description, task_context)
                    
                    results[agent_name] = agent_result
                    agent_outputs[agent_name] = agent_result
                    
                    # Check if agent failed and handle accordingly
                    if agent_result.get("status") == "error":
                        logger.warning(f"Agent {agent_name} failed: {agent_result.get('error', 'Unknown error')}")
                        
                        # Decide whether to continue or stop
                        if strategy == "multi_agent_sequential" and agent_name == "planner":
                            # If planner fails, we can't continue
                            break
                        # For other strategies, continue with remaining agents
            
            # Synthesize final result
            final_result = await self._synthesize_results(
                task_description, results, execution_plan
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in plan execution: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": results
            }
    
    def _build_agent_context(
        self,
        base_context: Dict[str, Any] = None,
        agent_outputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build context for agent execution"""
        
        context = base_context.copy() if base_context else {}
        
        if agent_outputs:
            context["previous_agent_outputs"] = agent_outputs
            
            # Extract key information from previous agents
            if "planner" in agent_outputs:
                planner_result = agent_outputs["planner"]
                if planner_result.get("status") == "completed":
                    # Try to extract plan from planner observations
                    observations = planner_result.get("observations", [])
                    for obs in observations:
                        if obs.get("metadata", {}).get("action_type") == "create_plan":
                            context["execution_plan"] = obs.get("content")
                            break
            
            if "researcher" in agent_outputs:
                researcher_result = agent_outputs["researcher"]
                if researcher_result.get("status") == "completed":
                    context["research_findings"] = researcher_result
        
        return context
    
    async def _synthesize_results(
        self,
        task_description: str,
        agent_results: Dict[str, Any],
        execution_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        
        # Determine overall success
        successful_agents = [
            name for name, result in agent_results.items()
            if result.get("status") == "completed"
        ]
        
        failed_agents = [
            name for name, result in agent_results.items()
            if result.get("status") in ["error", "failed"]
        ]
        
        overall_success = len(successful_agents) > 0 and len(failed_agents) == 0
        
        # Extract key outputs
        key_outputs = {}
        
        for agent_name, result in agent_results.items():
            if result.get("status") == "completed":
                # Extract final reflection or key observations
                if "final_reflection" in result:
                    key_outputs[f"{agent_name}_reflection"] = result["final_reflection"]
                
                # Extract successful observations
                observations = result.get("observations", [])
                successful_obs = [
                    obs for obs in observations
                    if obs.get("success", False)
                ]
                
                if successful_obs:
                    key_outputs[f"{agent_name}_outputs"] = successful_obs
        
        # Create synthesis
        synthesis = {
            "success": overall_success,
            "task_description": task_description,
            "execution_strategy": execution_plan.get("strategy"),
            "agents_used": list(agent_results.keys()),
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "key_outputs": key_outputs,
            "detailed_results": agent_results,
            "summary": self._create_result_summary(agent_results, overall_success)
        }
        
        return synthesis
    
    def _create_result_summary(
        self,
        agent_results: Dict[str, Any],
        overall_success: bool
    ) -> str:
        """Create a human-readable summary of results"""
        
        if overall_success:
            summary_parts = ["Task completed successfully."]
        else:
            summary_parts = ["Task completed with some issues."]
        
        for agent_name, result in agent_results.items():
            status = result.get("status", "unknown")
            
            if status == "completed":
                iterations = result.get("iterations", 0)
                summary_parts.append(f"{agent_name.title()} agent completed successfully in {iterations} iterations.")
            
            elif status == "error":
                error = result.get("error", "Unknown error")
                summary_parts.append(f"{agent_name.title()} agent failed: {error}")
            
            else:
                summary_parts.append(f"{agent_name.title()} agent status: {status}")
        
        return " ".join(summary_parts)
    
    async def _store_task_result(self, task_result: TaskResult):
        """Store task result in memory for future reference"""
        
        # Store task summary
        summary_content = f"""
Task: {task_result.task_description}
Status: {task_result.status}
Agents Used: {', '.join(task_result.agents_used)}
Execution Time: {task_result.execution_time:.2f}s
Summary: {task_result.result.get('summary', 'No summary available')}
"""
        
        await self.memory_manager.store_memory(
            content=summary_content,
            metadata={
                "type": "task_execution",
                "task_id": task_result.task_id,
                "status": task_result.status,
                "agents_used": task_result.agents_used,
                "execution_time": task_result.execution_time
            },
            category="task_history"
        )
        
        # Store detailed results if successful
        if task_result.status == "completed":
            await self.memory_manager.store_memory(
                content=json.dumps(task_result.result, indent=2),
                metadata={
                    "type": "task_result_detailed",
                    "task_id": task_result.task_id,
                    "task_description": task_result.task_description
                },
                category="task_results"
            )
    
    async def get_task_history(self, limit: int = 10) -> List[TaskResult]:
        """Get recent task execution history"""
        return self.task_history[-limit:]
    
    async def get_agent_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents"""
        
        status = {}
        
        for name, agent in self.agents.items():
            status[name] = {
                "name": agent.name,
                "role": agent.role,
                "state": agent.state.value,
                "iteration_count": agent.iteration_count,
                "session_memory_size": {
                    "thoughts": len(agent.session_memory.thoughts),
                    "actions": len(agent.session_memory.actions),
                    "observations": len(agent.session_memory.observations)
                }
            }
        
        return status
    
    def add_agent(self, name: str, agent: BaseAgent):
        """Add a new agent to the orchestrator"""
        self.agents[name] = agent
        logger.info(f"Added agent: {name}")
    
    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the orchestrator"""
        if name in self.agents:
            del self.agents[name]
            logger.info(f"Removed agent: {name}")
            return True
        return False


# Global orchestrator instance
orchestrator = AgentOrchestrator()


def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance"""
    return orchestrator