"""
Base Agent Architecture for HyperCortex-AI
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import structlog
from pydantic import BaseModel

from ..core.llm_engine import get_llm_engine, LLMResponse
from ..memory.vector_store import get_memory_manager, MemoryEntry
from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"


class AgentAction(BaseModel):
    """Represents an agent action"""
    action_type: str
    parameters: Dict[str, Any]
    reasoning: str
    confidence: float = 0.8


class AgentObservation(BaseModel):
    """Represents an observation from an action"""
    content: str
    success: bool
    metadata: Dict[str, Any] = {}


class AgentThought(BaseModel):
    """Represents an agent's thought process"""
    content: str
    reasoning_type: str  # "analysis", "planning", "reflection"
    confidence: float
    timestamp: datetime = datetime.utcnow()


class AgentMemory(BaseModel):
    """Agent's working memory for a session"""
    thoughts: List[AgentThought] = []
    actions: List[AgentAction] = []
    observations: List[AgentObservation] = []
    context: Dict[str, Any] = {}


class BaseAgent(ABC):
    """Base class for all HyperCortex agents"""
    
    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        tools: Optional[List[Any]] = None
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.tools = tools or []
        
        # Core components
        self.llm_engine = get_llm_engine()
        self.memory_manager = get_memory_manager()
        
        # Agent state
        self.state = AgentState.IDLE
        self.session_memory = AgentMemory()
        self.iteration_count = 0
        self.max_iterations = settings.max_iterations
        
        logger.info(f"Initialized agent: {self.name} ({self.role})")
    
    async def think(self, prompt: str, reasoning_type: str = "analysis") -> AgentThought:
        """Generate a thought using the LLM"""
        
        self.state = AgentState.THINKING
        
        # Build context from memory
        context = await self._build_context()
        
        # Create thinking prompt
        thinking_prompt = self._create_thinking_prompt(prompt, context)
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=thinking_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "type": "thinking"}
            )
            
            thought = AgentThought(
                content=response.content,
                reasoning_type=reasoning_type,
                confidence=0.8  # Could be extracted from response
            )
            
            self.session_memory.thoughts.append(thought)
            
            # Store important thoughts in long-term memory
            if reasoning_type in ["planning", "reflection"]:
                await self.memory_manager.store_memory(
                    content=thought.content,
                    metadata={
                        "agent": self.name,
                        "type": "thought",
                        "reasoning_type": reasoning_type
                    },
                    category=f"agent_{self.name}"
                )
            
            logger.info(f"Agent {self.name} generated thought: {reasoning_type}")
            return thought
            
        except Exception as e:
            logger.error(f"Error in thinking: {str(e)}")
            self.state = AgentState.ERROR
            raise e
    
    async def act(self, action: AgentAction) -> AgentObservation:
        """Execute an action"""
        
        self.state = AgentState.ACTING
        
        try:
            # Execute the action using available tools
            observation = await self._execute_action(action)
            
            # Store action and observation
            self.session_memory.actions.append(action)
            self.session_memory.observations.append(observation)
            
            # Store in long-term memory if significant
            if observation.success:
                await self.memory_manager.store_memory(
                    content=f"Action: {action.action_type} - Result: {observation.content}",
                    metadata={
                        "agent": self.name,
                        "type": "action_result",
                        "action_type": action.action_type,
                        "success": observation.success
                    },
                    category=f"agent_{self.name}"
                )
            
            logger.info(f"Agent {self.name} executed action: {action.action_type}")
            return observation
            
        except Exception as e:
            logger.error(f"Error in action execution: {str(e)}")
            observation = AgentObservation(
                content=f"Error executing action: {str(e)}",
                success=False,
                metadata={"error": str(e)}
            )
            self.session_memory.observations.append(observation)
            return observation
    
    async def reflect(self, context: str = "") -> AgentThought:
        """Reflect on recent actions and outcomes"""
        
        if not settings.reflection_enabled:
            return AgentThought(
                content="Reflection disabled",
                reasoning_type="reflection",
                confidence=1.0
            )
        
        self.state = AgentState.REFLECTING
        
        # Build reflection prompt
        reflection_prompt = self._create_reflection_prompt(context)
        
        reflection = await self.think(reflection_prompt, "reflection")
        
        logger.info(f"Agent {self.name} completed reflection")
        return reflection
    
    async def execute_task(self, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main execution loop for a task"""
        
        logger.info(f"Agent {self.name} starting task: {task}")
        
        # Initialize session
        self.session_memory = AgentMemory()
        self.session_memory.context = context or {}
        self.iteration_count = 0
        
        try:
            # Initial planning thought
            plan_thought = await self.think(
                f"Task: {task}\nContext: {json.dumps(context, indent=2) if context else 'None'}\n\nCreate a plan to accomplish this task.",
                "planning"
            )
            
            # Main execution loop
            while self.iteration_count < self.max_iterations:
                self.iteration_count += 1
                
                # Decide next action
                action = await self._decide_next_action(task)
                
                if action is None:
                    # Task completed
                    self.state = AgentState.COMPLETED
                    break
                
                # Execute action
                observation = await self.act(action)
                
                # Reflect if needed
                if self.iteration_count % 3 == 0:  # Reflect every 3 iterations
                    await self.reflect(f"Task progress for: {task}")
                
                # Check if task is complete
                if await self._is_task_complete(task, observation):
                    self.state = AgentState.COMPLETED
                    break
            
            # Final reflection
            final_reflection = await self.reflect(f"Task completion for: {task}")
            
            # Prepare result
            result = {
                "task": task,
                "status": self.state.value,
                "iterations": self.iteration_count,
                "thoughts": [t.dict() for t in self.session_memory.thoughts],
                "actions": [a.dict() for a in self.session_memory.actions],
                "observations": [o.dict() for o in self.session_memory.observations],
                "final_reflection": final_reflection.content
            }
            
            logger.info(f"Agent {self.name} completed task with status: {self.state.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error in task execution: {str(e)}")
            self.state = AgentState.ERROR
            return {
                "task": task,
                "status": "error",
                "error": str(e),
                "iterations": self.iteration_count
            }
    
    # Abstract methods to be implemented by specific agents
    
    @abstractmethod
    async def _decide_next_action(self, task: str) -> Optional[AgentAction]:
        """Decide the next action to take"""
        pass
    
    @abstractmethod
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute a specific action"""
        pass
    
    @abstractmethod
    async def _is_task_complete(self, task: str, last_observation: AgentObservation) -> bool:
        """Check if the task is complete"""
        pass
    
    # Helper methods
    
    async def _build_context(self) -> str:
        """Build context from memory and session"""
        
        # Get relevant memories
        if self.session_memory.thoughts:
            last_thought = self.session_memory.thoughts[-1].content
            relevant_memories = await self.memory_manager.recall_memories(
                last_thought,
                k=3,
                category=f"agent_{self.name}"
            )
        else:
            relevant_memories = []
        
        # Build context string
        context_parts = []
        
        if relevant_memories:
            context_parts.append("Relevant past experiences:")
            for memory, score in relevant_memories:
                context_parts.append(f"- {memory.content} (relevance: {score:.2f})")
        
        if self.session_memory.thoughts:
            context_parts.append("\nRecent thoughts:")
            for thought in self.session_memory.thoughts[-3:]:
                context_parts.append(f"- {thought.content}")
        
        if self.session_memory.observations:
            context_parts.append("\nRecent observations:")
            for obs in self.session_memory.observations[-3:]:
                context_parts.append(f"- {obs.content}")
        
        return "\n".join(context_parts)
    
    def _create_thinking_prompt(self, prompt: str, context: str) -> str:
        """Create a structured thinking prompt"""
        
        return f"""
{context}

Current situation: {prompt}

Think step by step about this situation. Consider:
1. What do I understand about the current state?
2. What are my options?
3. What would be the best approach?
4. What potential issues should I consider?

Provide your analysis and reasoning:
"""
    
    def _create_reflection_prompt(self, context: str) -> str:
        """Create a reflection prompt"""
        
        recent_actions = self.session_memory.actions[-3:] if self.session_memory.actions else []
        recent_observations = self.session_memory.observations[-3:] if self.session_memory.observations else []
        
        return f"""
{context}

Recent actions taken:
{json.dumps([a.dict() for a in recent_actions], indent=2)}

Recent observations:
{json.dumps([o.dict() for o in recent_observations], indent=2)}

Reflect on:
1. What worked well in my recent actions?
2. What could have been done better?
3. What did I learn from the observations?
4. How should I adjust my approach going forward?

Provide your reflection:
"""