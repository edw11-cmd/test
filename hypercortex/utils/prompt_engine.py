"""
Prompt Engineering Engine for HyperCortex-AI
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import structlog

from ..core.llm_engine import get_llm_engine
from ..memory.vector_store import get_memory_manager
from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class PromptTemplate:
    """Represents a prompt template"""
    
    def __init__(
        self,
        name: str,
        template: str,
        variables: List[str],
        description: str = "",
        category: str = "general"
    ):
        self.name = name
        self.template = template
        self.variables = variables
        self.description = description
        self.category = category
        self.usage_count = 0
        self.success_rate = 0.0
        self.created_at = datetime.utcnow()
    
    def render(self, **kwargs) -> str:
        """Render the template with provided variables"""
        
        # Check if all required variables are provided
        missing_vars = [var for var in self.variables if var not in kwargs]
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")
        
        try:
            rendered = self.template.format(**kwargs)
            self.usage_count += 1
            return rendered
        except KeyError as e:
            raise ValueError(f"Template variable not found: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "description": self.description,
            "category": self.category,
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat()
        }


class PromptOptimizer:
    """Optimizes prompts for better performance"""
    
    def __init__(self):
        self.llm_engine = get_llm_engine()
        self.memory_manager = get_memory_manager()
        
        # Optimization strategies
        self.optimization_strategies = [
            "clarity_improvement",
            "specificity_enhancement",
            "context_enrichment",
            "instruction_refinement",
            "example_addition"
        ]
    
    async def optimize_prompt(
        self,
        original_prompt: str,
        task_context: str,
        performance_feedback: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Optimize a prompt for better performance"""
        
        optimization_prompt = f"""
You are a prompt engineering expert. Your task is to optimize the following prompt for better performance.

Original Prompt:
{original_prompt}

Task Context:
{task_context}

Performance Feedback (if any):
{performance_feedback or "No specific feedback provided"}

Optimize the prompt by:
1. Improving clarity and specificity
2. Adding relevant context and examples
3. Structuring instructions more effectively
4. Removing ambiguity and redundancy
5. Enhancing the prompt's ability to elicit the desired response

Provide your optimization in the following JSON format:
{{
    "optimized_prompt": "the improved prompt",
    "changes_made": ["list of specific changes"],
    "reasoning": "explanation of why these changes improve the prompt",
    "expected_improvements": ["list of expected performance improvements"],
    "optimization_strategy": "primary strategy used",
    "confidence_score": 0.85
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=optimization_prompt,
                system_message="You are an expert prompt engineer focused on creating high-performance prompts.",
                metadata={"type": "prompt_optimization"}
            )
            
            optimization_data = json.loads(response.content)
            
            # Store optimization in memory
            await self.memory_manager.store_memory(
                content=f"Prompt optimization: {original_prompt} -> {optimization_data['optimized_prompt']}",
                metadata={
                    "type": "prompt_optimization",
                    "original_length": len(original_prompt),
                    "optimized_length": len(optimization_data["optimized_prompt"]),
                    "strategy": optimization_data.get("optimization_strategy"),
                    "confidence": optimization_data.get("confidence_score", 0.0)
                },
                category="prompt_optimizations"
            )
            
            return optimization_data["optimized_prompt"], optimization_data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse optimization response as JSON")
            return original_prompt, {"error": "Failed to optimize prompt"}
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return original_prompt, {"error": str(e)}
    
    async def generate_prompt_variants(
        self,
        base_prompt: str,
        num_variants: int = 3
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate multiple variants of a prompt"""
        
        variant_prompt = f"""
Create {num_variants} different variants of the following prompt, each optimized for different aspects:

Base Prompt:
{base_prompt}

For each variant, focus on a different optimization strategy:
1. Clarity and directness
2. Detailed context and examples
3. Step-by-step instructions

Provide the variants in JSON format:
{{
    "variants": [
        {{
            "prompt": "variant 1",
            "focus": "clarity and directness",
            "description": "what makes this variant unique"
        }},
        {{
            "prompt": "variant 2", 
            "focus": "detailed context",
            "description": "what makes this variant unique"
        }},
        {{
            "prompt": "variant 3",
            "focus": "step-by-step instructions", 
            "description": "what makes this variant unique"
        }}
    ]
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=variant_prompt,
                system_message="You are a prompt engineering expert creating diverse, high-quality prompt variants.",
                metadata={"type": "prompt_variants"}
            )
            
            variants_data = json.loads(response.content)
            
            results = []
            for variant in variants_data.get("variants", []):
                results.append((
                    variant["prompt"],
                    {
                        "focus": variant["focus"],
                        "description": variant["description"],
                        "variant_type": "generated"
                    }
                ))
            
            return results
            
        except json.JSONDecodeError:
            logger.error("Failed to parse variants response as JSON")
            return [(base_prompt, {"error": "Failed to generate variants"})]
        except Exception as e:
            logger.error(f"Error generating prompt variants: {str(e)}")
            return [(base_prompt, {"error": str(e)})]


class PromptEngine:
    """Main prompt engineering engine"""
    
    def __init__(self):
        self.llm_engine = get_llm_engine()
        self.memory_manager = get_memory_manager()
        self.optimizer = PromptOptimizer()
        
        # Template storage
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Performance tracking
        self.prompt_performance: Dict[str, Dict[str, Any]] = {}
        
        # Load built-in templates
        self._load_builtin_templates()
        
        logger.info("Prompt Engine initialized")
    
    def _load_builtin_templates(self):
        """Load built-in prompt templates"""
        
        # Agent system prompts
        self.register_template(PromptTemplate(
            name="planner_system",
            template="""You are a strategic planning agent in the HyperCortex-AI system. Your role is to:

1. Break down complex tasks into manageable subtasks
2. Create detailed execution plans with dependencies
3. Identify required resources and tools
4. Anticipate potential challenges and create contingency plans
5. Coordinate with other agents in the system

You think systematically and consider multiple approaches before deciding on the best plan.
Always provide clear, actionable steps with success criteria.

Current context: {context}
Task complexity: {complexity}
Available resources: {resources}""",
            variables=["context", "complexity", "resources"],
            description="System prompt for the planner agent",
            category="agent_system"
        ))
        
        self.register_template(PromptTemplate(
            name="research_system",
            template="""You are a research specialist in the HyperCortex-AI system. Your role is to:

1. Gather comprehensive information on topics and questions
2. Analyze and synthesize information from multiple sources
3. Identify key insights, trends, and patterns
4. Provide evidence-based conclusions and recommendations
5. Fact-check and verify information accuracy

You approach research systematically and critically evaluate sources for credibility and relevance.

Research scope: {scope}
Information sources: {sources}
Quality standards: {quality_standards}""",
            variables=["scope", "sources", "quality_standards"],
            description="System prompt for the research agent",
            category="agent_system"
        ))
        
        # Task execution templates
        self.register_template(PromptTemplate(
            name="task_analysis",
            template="""Analyze the following task and provide a structured breakdown:

Task: {task}
Context: {context}

Provide analysis in the following format:
1. Task Type: [classification]
2. Complexity Level: [low/medium/high]
3. Required Skills: [list of skills needed]
4. Estimated Duration: [time estimate]
5. Key Challenges: [potential difficulties]
6. Success Criteria: [how to measure success]
7. Recommended Approach: [suggested strategy]

Analysis:""",
            variables=["task", "context"],
            description="Template for analyzing tasks before execution",
            category="task_execution"
        ))
        
        # Reflection templates
        self.register_template(PromptTemplate(
            name="performance_reflection",
            template="""Reflect on the following performance data and provide insights:

Task: {task}
Actions Taken: {actions}
Results: {results}
Performance Metrics: {metrics}

Reflection Questions:
1. What worked well in this execution?
2. What could have been done better?
3. What patterns do you notice in the performance?
4. What lessons can be learned for future tasks?
5. How should the approach be adjusted going forward?

Provide your reflection:""",
            variables=["task", "actions", "results", "metrics"],
            description="Template for performance reflection",
            category="reflection"
        ))
    
    def register_template(self, template: PromptTemplate):
        """Register a new prompt template"""
        self.templates[template.name] = template
        logger.info(f"Registered prompt template: {template.name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name"""
        return self.templates.get(name)
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """List available templates, optionally filtered by category"""
        if category:
            return [
                name for name, template in self.templates.items()
                if template.category == category
            ]
        return list(self.templates.keys())
    
    def render_template(self, name: str, **kwargs) -> str:
        """Render a template with provided variables"""
        template = self.get_template(name)
        if not template:
            raise ValueError(f"Template '{name}' not found")
        
        return template.render(**kwargs)
    
    async def optimize_template(
        self,
        template_name: str,
        performance_feedback: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Optimize an existing template"""
        
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        optimized_prompt, optimization_data = await self.optimizer.optimize_prompt(
            original_prompt=template.template,
            task_context=f"Template: {template.description}, Category: {template.category}",
            performance_feedback=performance_feedback
        )
        
        # Create optimized template
        optimized_template = PromptTemplate(
            name=f"{template_name}_optimized",
            template=optimized_prompt,
            variables=template.variables,
            description=f"Optimized version of {template.description}",
            category=template.category
        )
        
        self.register_template(optimized_template)
        
        return optimized_prompt, optimization_data
    
    async def generate_dynamic_prompt(
        self,
        task_description: str,
        agent_role: str,
        context: Dict[str, Any] = None
    ) -> str:
        """Generate a dynamic prompt for a specific task and agent"""
        
        generation_prompt = f"""
Generate an optimized prompt for the following scenario:

Task Description: {task_description}
Agent Role: {agent_role}
Context: {json.dumps(context, indent=2) if context else "No additional context"}

The prompt should:
1. Be clear and specific to the task
2. Include relevant context and constraints
3. Provide clear instructions for the agent
4. Include success criteria
5. Be optimized for the agent's role and capabilities

Generate the prompt:"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=generation_prompt,
                system_message="You are an expert prompt engineer creating task-specific prompts for AI agents.",
                metadata={"type": "dynamic_prompt_generation"}
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating dynamic prompt: {str(e)}")
            return f"Complete the following task: {task_description}"
    
    def track_prompt_performance(
        self,
        prompt_id: str,
        success: bool,
        execution_time: float,
        quality_score: Optional[float] = None
    ):
        """Track performance of a prompt"""
        
        if prompt_id not in self.prompt_performance:
            self.prompt_performance[prompt_id] = {
                "usage_count": 0,
                "success_count": 0,
                "total_execution_time": 0.0,
                "quality_scores": []
            }
        
        perf = self.prompt_performance[prompt_id]
        perf["usage_count"] += 1
        perf["total_execution_time"] += execution_time
        
        if success:
            perf["success_count"] += 1
        
        if quality_score is not None:
            perf["quality_scores"].append(quality_score)
        
        # Update template success rate if it's a template
        if prompt_id in self.templates:
            template = self.templates[prompt_id]
            template.success_rate = perf["success_count"] / perf["usage_count"]
    
    def get_prompt_analytics(self, prompt_id: str) -> Dict[str, Any]:
        """Get analytics for a specific prompt"""
        
        if prompt_id not in self.prompt_performance:
            return {"error": "No performance data available"}
        
        perf = self.prompt_performance[prompt_id]
        
        analytics = {
            "usage_count": perf["usage_count"],
            "success_rate": perf["success_count"] / perf["usage_count"],
            "average_execution_time": perf["total_execution_time"] / perf["usage_count"],
            "total_execution_time": perf["total_execution_time"]
        }
        
        if perf["quality_scores"]:
            analytics["average_quality_score"] = sum(perf["quality_scores"]) / len(perf["quality_scores"])
            analytics["quality_score_trend"] = perf["quality_scores"][-10:]  # Last 10 scores
        
        return analytics
    
    async def suggest_prompt_improvements(
        self,
        prompt_id: str
    ) -> List[Dict[str, Any]]:
        """Suggest improvements for a prompt based on performance data"""
        
        analytics = self.get_prompt_analytics(prompt_id)
        
        if "error" in analytics:
            return []
        
        suggestions = []
        
        # Low success rate
        if analytics["success_rate"] < 0.7:
            suggestions.append({
                "type": "success_rate",
                "issue": "Low success rate",
                "suggestion": "Consider optimizing for clarity and specificity",
                "priority": "high"
            })
        
        # Slow execution
        if analytics["average_execution_time"] > 5.0:
            suggestions.append({
                "type": "performance",
                "issue": "Slow execution time",
                "suggestion": "Consider simplifying the prompt or breaking it into smaller parts",
                "priority": "medium"
            })
        
        # Quality issues
        if "average_quality_score" in analytics and analytics["average_quality_score"] < 0.6:
            suggestions.append({
                "type": "quality",
                "issue": "Low quality scores",
                "suggestion": "Consider adding more context and examples",
                "priority": "high"
            })
        
        return suggestions


# Global prompt engine instance
prompt_engine = PromptEngine()


def get_prompt_engine() -> PromptEngine:
    """Get the global prompt engine instance"""
    return prompt_engine