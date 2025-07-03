"""
Planner Agent - Strategic planning and task decomposition
"""

import json
from typing import Dict, List, Any, Optional
import structlog

from .base_agent import BaseAgent, AgentAction, AgentObservation
from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class PlannerAgent(BaseAgent):
    """Agent specialized in strategic planning and task decomposition"""
    
    def __init__(self):
        system_prompt = """You are a strategic planning agent in the HyperCortex-AI system. Your role is to:

1. Break down complex tasks into manageable subtasks
2. Create detailed execution plans with dependencies
3. Identify required resources and tools
4. Anticipate potential challenges and create contingency plans
5. Coordinate with other agents in the system

You think systematically and consider multiple approaches before deciding on the best plan.
Always provide clear, actionable steps with success criteria.

When planning, consider:
- Task complexity and scope
- Available resources and tools
- Time constraints
- Risk factors
- Dependencies between subtasks
- Success metrics

Format your plans as structured JSON with clear steps, dependencies, and success criteria."""

        super().__init__(
            name="PlannerAgent",
            role="Strategic Planner",
            system_prompt=system_prompt
        )
    
    async def _decide_next_action(self, task: str) -> Optional[AgentAction]:
        """Decide the next planning action"""
        
        # Check if we already have a plan
        existing_plans = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "create_plan" and obs.success
        ]
        
        if not existing_plans:
            # Create initial plan
            return AgentAction(
                action_type="create_plan",
                parameters={"task": task},
                reasoning="Need to create an initial plan for the task",
                confidence=0.9
            )
        
        # Check if plan needs refinement
        last_observation = self.session_memory.observations[-1] if self.session_memory.observations else None
        
        if last_observation and not last_observation.success:
            return AgentAction(
                action_type="refine_plan",
                parameters={"task": task, "previous_attempt": last_observation.content},
                reasoning="Previous plan had issues, need to refine",
                confidence=0.8
            )
        
        # Check if we need to validate the plan
        validation_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "validate_plan"
        ]
        
        if not validation_actions and existing_plans:
            return AgentAction(
                action_type="validate_plan",
                parameters={"task": task},
                reasoning="Need to validate the created plan",
                confidence=0.8
            )
        
        # Planning is complete
        return None
    
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute planning actions"""
        
        if action.action_type == "create_plan":
            return await self._create_plan(action.parameters["task"])
        
        elif action.action_type == "refine_plan":
            return await self._refine_plan(
                action.parameters["task"],
                action.parameters.get("previous_attempt", "")
            )
        
        elif action.action_type == "validate_plan":
            return await self._validate_plan(action.parameters["task"])
        
        else:
            return AgentObservation(
                content=f"Unknown action type: {action.action_type}",
                success=False,
                metadata={"action_type": action.action_type}
            )
    
    async def _is_task_complete(self, task: str, last_observation: AgentObservation) -> bool:
        """Check if planning is complete"""
        
        # Planning is complete when we have a validated plan
        validated_plans = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "validate_plan" and obs.success
        ]
        
        return len(validated_plans) > 0
    
    async def _create_plan(self, task: str) -> AgentObservation:
        """Create a detailed plan for the task"""
        
        planning_prompt = f"""
Task to plan: {task}

Create a comprehensive execution plan with the following structure:

{{
    "task_analysis": {{
        "complexity": "low|medium|high",
        "estimated_duration": "time estimate",
        "required_skills": ["skill1", "skill2"],
        "success_criteria": ["criteria1", "criteria2"]
    }},
    "subtasks": [
        {{
            "id": "subtask_1",
            "title": "Subtask title",
            "description": "Detailed description",
            "dependencies": ["subtask_id"],
            "estimated_duration": "time estimate",
            "required_tools": ["tool1", "tool2"],
            "success_criteria": ["criteria1"],
            "assigned_agent": "agent_type"
        }}
    ],
    "execution_order": ["subtask_1", "subtask_2"],
    "risk_assessment": {{
        "potential_risks": ["risk1", "risk2"],
        "mitigation_strategies": ["strategy1", "strategy2"]
    }},
    "resource_requirements": {{
        "tools": ["tool1", "tool2"],
        "external_apis": ["api1"],
        "data_sources": ["source1"]
    }}
}}

Provide a detailed, actionable plan:
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=planning_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "create_plan"}
            )
            
            # Try to parse as JSON to validate structure
            try:
                plan_data = json.loads(response.content)
                
                # Store the plan in memory
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "plan",
                        "task": task,
                        "plan_version": 1
                    },
                    category="plans"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "create_plan",
                        "plan_structure_valid": True,
                        "subtask_count": len(plan_data.get("subtasks", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Plan created but not in valid JSON format: {response.content}",
                    success=False,
                    metadata={"action_type": "create_plan", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}")
            return AgentObservation(
                content=f"Error creating plan: {str(e)}",
                success=False,
                metadata={"action_type": "create_plan", "error": str(e)}
            )
    
    async def _refine_plan(self, task: str, previous_attempt: str) -> AgentObservation:
        """Refine an existing plan based on feedback"""
        
        refinement_prompt = f"""
Original task: {task}

Previous plan attempt:
{previous_attempt}

The previous plan had issues. Please create an improved version that addresses the problems.
Consider:
1. What went wrong with the previous plan?
2. How can the plan be made more robust?
3. Are there missing steps or dependencies?
4. Can the plan be simplified or made more efficient?

Provide the refined plan in the same JSON structure:
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=refinement_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "refine_plan"}
            )
            
            # Validate JSON structure
            try:
                plan_data = json.loads(response.content)
                
                # Store the refined plan
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "plan",
                        "task": task,
                        "plan_version": 2,
                        "refinement": True
                    },
                    category="plans"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "refine_plan",
                        "plan_structure_valid": True,
                        "subtask_count": len(plan_data.get("subtasks", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Refined plan not in valid JSON format: {response.content}",
                    success=False,
                    metadata={"action_type": "refine_plan", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error refining plan: {str(e)}")
            return AgentObservation(
                content=f"Error refining plan: {str(e)}",
                success=False,
                metadata={"action_type": "refine_plan", "error": str(e)}
            )
    
    async def _validate_plan(self, task: str) -> AgentObservation:
        """Validate the created plan"""
        
        # Get the latest plan
        plan_observations = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") in ["create_plan", "refine_plan"] and obs.success
        ]
        
        if not plan_observations:
            return AgentObservation(
                content="No plan found to validate",
                success=False,
                metadata={"action_type": "validate_plan", "error": "no_plan"}
            )
        
        latest_plan = plan_observations[-1].content
        
        validation_prompt = f"""
Task: {task}

Plan to validate:
{latest_plan}

Validate this plan by checking:
1. Completeness: Does the plan cover all aspects of the task?
2. Feasibility: Are all subtasks realistic and achievable?
3. Dependencies: Are dependencies correctly identified and ordered?
4. Resource requirements: Are all necessary resources identified?
5. Success criteria: Are success criteria clear and measurable?
6. Risk assessment: Are potential risks and mitigations adequate?

Provide a validation report with:
- Overall assessment (valid/needs_improvement)
- Specific issues found (if any)
- Recommendations for improvement
- Confidence score (0-1)

Format as JSON:
{{
    "validation_result": "valid|needs_improvement",
    "confidence_score": 0.85,
    "issues_found": ["issue1", "issue2"],
    "recommendations": ["rec1", "rec2"],
    "overall_assessment": "detailed assessment"
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=validation_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "validate_plan"}
            )
            
            # Try to parse validation result
            try:
                validation_data = json.loads(response.content)
                is_valid = validation_data.get("validation_result") == "valid"
                
                return AgentObservation(
                    content=response.content,
                    success=is_valid,
                    metadata={
                        "action_type": "validate_plan",
                        "validation_result": validation_data.get("validation_result"),
                        "confidence_score": validation_data.get("confidence_score", 0.0)
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Validation completed but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "validate_plan", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error validating plan: {str(e)}")
            return AgentObservation(
                content=f"Error validating plan: {str(e)}",
                success=False,
                metadata={"action_type": "validate_plan", "error": str(e)}
            )
    
    async def get_latest_plan(self) -> Optional[Dict[str, Any]]:
        """Get the latest validated plan"""
        
        plan_observations = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") in ["create_plan", "refine_plan"] and obs.success
        ]
        
        if not plan_observations:
            return None
        
        try:
            latest_plan_content = plan_observations[-1].content
            return json.loads(latest_plan_content)
        except json.JSONDecodeError:
            return None