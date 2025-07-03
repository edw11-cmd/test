"""
Coder Agent - Code generation, analysis, and implementation
"""

import json
import re
from typing import Dict, List, Any, Optional
import structlog

from .base_agent import BaseAgent, AgentAction, AgentObservation
from ..core.config import get_settings

logger = structlog.get_logger(__name__)
settings = get_settings()


class CoderAgent(BaseAgent):
    """Agent specialized in code generation, analysis, and implementation"""
    
    def __init__(self):
        system_prompt = """You are an expert software engineer in the HyperCortex-AI system. Your role is to:

1. Write high-quality, efficient, and maintainable code
2. Analyze existing codebases and identify improvements
3. Debug and fix code issues
4. Implement features based on specifications
5. Follow best practices and coding standards
6. Write comprehensive tests and documentation

You are proficient in multiple programming languages including Python, JavaScript, TypeScript, Go, Rust, and others.
You always consider security, performance, and maintainability in your code.

When writing code:
- Follow clean code principles
- Include proper error handling
- Add meaningful comments and docstrings
- Consider edge cases and validation
- Write testable and modular code
- Follow language-specific best practices

When analyzing code:
- Identify potential bugs and security issues
- Suggest performance improvements
- Check for code smells and anti-patterns
- Verify adherence to best practices
- Assess code maintainability and readability"""

        super().__init__(
            name="CoderAgent",
            role="Software Engineer",
            system_prompt=system_prompt
        )
    
    async def _decide_next_action(self, task: str) -> Optional[AgentAction]:
        """Decide the next coding action"""
        
        # Analyze the task to determine what type of coding work is needed
        task_lower = task.lower()
        
        # Check if we need to analyze requirements first
        requirements_analysis = [
            action for action in self.session_memory.actions
            if action.action_type == "analyze_requirements"
        ]
        
        if not requirements_analysis:
            return AgentAction(
                action_type="analyze_requirements",
                parameters={"task": task},
                reasoning="Need to analyze requirements before coding",
                confidence=0.9
            )
        
        # Check if we need to design the solution
        design_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "design_solution"
        ]
        
        if not design_actions:
            return AgentAction(
                action_type="design_solution",
                parameters={"task": task},
                reasoning="Need to design the solution architecture",
                confidence=0.8
            )
        
        # Check if we need to implement code
        implementation_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "implement_code"
        ]
        
        if not implementation_actions:
            return AgentAction(
                action_type="implement_code",
                parameters={"task": task},
                reasoning="Ready to implement the code solution",
                confidence=0.8
            )
        
        # Check if we need to review/test the code
        review_actions = [
            action for action in self.session_memory.actions
            if action.action_type == "review_code"
        ]
        
        if not review_actions and implementation_actions:
            return AgentAction(
                action_type="review_code",
                parameters={"task": task},
                reasoning="Need to review and test the implemented code",
                confidence=0.8
            )
        
        # Check if we need to fix any issues
        last_observation = self.session_memory.observations[-1] if self.session_memory.observations else None
        
        if last_observation and not last_observation.success and "error" in last_observation.content.lower():
            return AgentAction(
                action_type="fix_issues",
                parameters={"task": task, "issues": last_observation.content},
                reasoning="Need to fix identified issues",
                confidence=0.7
            )
        
        # Coding task is complete
        return None
    
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute coding actions"""
        
        if action.action_type == "analyze_requirements":
            return await self._analyze_requirements(action.parameters["task"])
        
        elif action.action_type == "design_solution":
            return await self._design_solution(action.parameters["task"])
        
        elif action.action_type == "implement_code":
            return await self._implement_code(action.parameters["task"])
        
        elif action.action_type == "review_code":
            return await self._review_code(action.parameters["task"])
        
        elif action.action_type == "fix_issues":
            return await self._fix_issues(
                action.parameters["task"],
                action.parameters.get("issues", "")
            )
        
        else:
            return AgentObservation(
                content=f"Unknown action type: {action.action_type}",
                success=False,
                metadata={"action_type": action.action_type}
            )
    
    async def _is_task_complete(self, task: str, last_observation: AgentObservation) -> bool:
        """Check if coding task is complete"""
        
        # Task is complete when we have successfully reviewed code
        review_actions = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "review_code" and obs.success
        ]
        
        return len(review_actions) > 0
    
    async def _analyze_requirements(self, task: str) -> AgentObservation:
        """Analyze coding requirements"""
        
        analysis_prompt = f"""
Coding task: {task}

Analyze this coding task and provide a detailed requirements analysis:

1. **Functional Requirements**:
   - What specific functionality needs to be implemented?
   - What are the inputs and expected outputs?
   - What are the core features and capabilities?

2. **Technical Requirements**:
   - What programming language(s) should be used?
   - What frameworks, libraries, or dependencies are needed?
   - What are the performance requirements?
   - What are the security considerations?

3. **Architecture Considerations**:
   - What design patterns should be used?
   - How should the code be structured?
   - What are the key components and their interactions?

4. **Constraints and Assumptions**:
   - What are the limitations or constraints?
   - What assumptions are being made?
   - What edge cases need to be considered?

5. **Success Criteria**:
   - How will we know the implementation is successful?
   - What tests need to be written?
   - What documentation is required?

Provide a comprehensive analysis in JSON format:
{{
    "functional_requirements": ["req1", "req2"],
    "technical_requirements": {{
        "language": "python",
        "frameworks": ["fastapi", "pydantic"],
        "dependencies": ["dep1", "dep2"],
        "performance": "requirements",
        "security": "considerations"
    }},
    "architecture": {{
        "patterns": ["pattern1", "pattern2"],
        "structure": "description",
        "components": ["comp1", "comp2"]
    }},
    "constraints": ["constraint1", "constraint2"],
    "success_criteria": ["criteria1", "criteria2"],
    "estimated_complexity": "low|medium|high"
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=analysis_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "analyze_requirements"}
            )
            
            # Try to parse as JSON
            try:
                requirements_data = json.loads(response.content)
                
                # Store requirements in memory
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "requirements",
                        "task": task,
                        "complexity": requirements_data.get("estimated_complexity", "medium")
                    },
                    category="coding_requirements"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "analyze_requirements",
                        "complexity": requirements_data.get("estimated_complexity", "medium"),
                        "language": requirements_data.get("technical_requirements", {}).get("language", "unknown")
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Requirements analyzed but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "analyze_requirements", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error analyzing requirements: {str(e)}")
            return AgentObservation(
                content=f"Error analyzing requirements: {str(e)}",
                success=False,
                metadata={"action_type": "analyze_requirements", "error": str(e)}
            )
    
    async def _design_solution(self, task: str) -> AgentObservation:
        """Design the code solution"""
        
        # Get requirements from previous analysis
        requirements_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "analyze_requirements" and obs.success
        ]
        
        requirements = requirements_obs[-1].content if requirements_obs else "No requirements analysis available"
        
        design_prompt = f"""
Coding task: {task}

Requirements analysis:
{requirements}

Design a detailed solution for this coding task:

1. **High-Level Architecture**:
   - Overall system design
   - Key components and their responsibilities
   - Data flow and interactions

2. **Detailed Design**:
   - Class/function structure
   - API design (if applicable)
   - Database schema (if applicable)
   - File structure and organization

3. **Implementation Plan**:
   - Step-by-step implementation approach
   - Order of development
   - Key milestones

4. **Code Structure**:
   - Main files and modules
   - Key classes and functions
   - Configuration and setup

5. **Testing Strategy**:
   - Unit tests to write
   - Integration tests needed
   - Test data requirements

Provide the design in JSON format:
{{
    "architecture": {{
        "overview": "description",
        "components": [
            {{
                "name": "component_name",
                "responsibility": "what it does",
                "interfaces": ["interface1", "interface2"]
            }}
        ],
        "data_flow": "description"
    }},
    "implementation_plan": [
        {{
            "step": 1,
            "description": "what to implement",
            "files": ["file1.py", "file2.py"],
            "estimated_effort": "time estimate"
        }}
    ],
    "file_structure": {{
        "main_files": ["file1.py", "file2.py"],
        "test_files": ["test_file1.py"],
        "config_files": ["config.py"]
    }},
    "testing_strategy": {{
        "unit_tests": ["test1", "test2"],
        "integration_tests": ["test1"],
        "test_data": "requirements"
    }}
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=design_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "design_solution"}
            )
            
            # Try to parse as JSON
            try:
                design_data = json.loads(response.content)
                
                # Store design in memory
                await self.memory_manager.store_memory(
                    content=response.content,
                    metadata={
                        "agent": self.name,
                        "type": "design",
                        "task": task,
                        "component_count": len(design_data.get("architecture", {}).get("components", []))
                    },
                    category="coding_designs"
                )
                
                return AgentObservation(
                    content=response.content,
                    success=True,
                    metadata={
                        "action_type": "design_solution",
                        "component_count": len(design_data.get("architecture", {}).get("components", [])),
                        "file_count": len(design_data.get("file_structure", {}).get("main_files", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Solution designed but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "design_solution", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error designing solution: {str(e)}")
            return AgentObservation(
                content=f"Error designing solution: {str(e)}",
                success=False,
                metadata={"action_type": "design_solution", "error": str(e)}
            )
    
    async def _implement_code(self, task: str) -> AgentObservation:
        """Implement the code solution"""
        
        # Get design from previous step
        design_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "design_solution" and obs.success
        ]
        
        design = design_obs[-1].content if design_obs else "No design available"
        
        implementation_prompt = f"""
Coding task: {task}

Solution design:
{design}

Implement the complete code solution based on the design. Provide:

1. **Complete Code Implementation**:
   - All necessary files with full code
   - Proper imports and dependencies
   - Error handling and validation
   - Comments and docstrings

2. **Configuration and Setup**:
   - Configuration files
   - Requirements/dependencies
   - Setup instructions

3. **Example Usage**:
   - How to use the implemented code
   - Example inputs and outputs
   - API usage examples (if applicable)

Format the response as:
```
FILE: filename.py
```python
# Complete code here
```

FILE: another_file.py
```python
# More code here
```

SETUP:
```bash
# Setup commands
```

USAGE:
```python
# Usage examples
```
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=implementation_prompt,
                system_message=self.system_prompt,
                max_tokens=4000,  # Allow for longer code responses
                metadata={"agent": self.name, "action": "implement_code"}
            )
            
            # Extract code files from response
            code_files = self._extract_code_files(response.content)
            
            # Store implementation in memory
            await self.memory_manager.store_memory(
                content=response.content,
                metadata={
                    "agent": self.name,
                    "type": "implementation",
                    "task": task,
                    "file_count": len(code_files),
                    "total_lines": sum(len(code.split('\n')) for code in code_files.values())
                },
                category="code_implementations"
            )
            
            return AgentObservation(
                content=response.content,
                success=True,
                metadata={
                    "action_type": "implement_code",
                    "file_count": len(code_files),
                    "total_lines": sum(len(code.split('\n')) for code in code_files.values())
                }
            )
            
        except Exception as e:
            logger.error(f"Error implementing code: {str(e)}")
            return AgentObservation(
                content=f"Error implementing code: {str(e)}",
                success=False,
                metadata={"action_type": "implement_code", "error": str(e)}
            )
    
    async def _review_code(self, task: str) -> AgentObservation:
        """Review the implemented code"""
        
        # Get implementation from previous step
        implementation_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "implement_code" and obs.success
        ]
        
        if not implementation_obs:
            return AgentObservation(
                content="No implementation found to review",
                success=False,
                metadata={"action_type": "review_code", "error": "no_implementation"}
            )
        
        implementation = implementation_obs[-1].content
        
        review_prompt = f"""
Coding task: {task}

Implementation to review:
{implementation}

Perform a comprehensive code review checking:

1. **Code Quality**:
   - Clean code principles
   - Readability and maintainability
   - Proper naming conventions
   - Code organization and structure

2. **Functionality**:
   - Does the code meet the requirements?
   - Are all edge cases handled?
   - Is error handling adequate?
   - Are inputs properly validated?

3. **Security**:
   - Are there any security vulnerabilities?
   - Is input sanitization proper?
   - Are secrets handled securely?

4. **Performance**:
   - Are there any performance issues?
   - Can the code be optimized?
   - Are resources used efficiently?

5. **Testing**:
   - Is the code testable?
   - Are there obvious test cases missing?
   - Is test coverage adequate?

6. **Documentation**:
   - Are docstrings and comments adequate?
   - Is the code self-documenting?
   - Are usage examples clear?

Provide review results in JSON format:
{{
    "overall_quality": "excellent|good|fair|poor",
    "functionality_score": 0.85,
    "security_score": 0.90,
    "performance_score": 0.80,
    "issues_found": [
        {{
            "severity": "high|medium|low",
            "category": "security|performance|functionality|style",
            "description": "issue description",
            "suggestion": "how to fix"
        }}
    ],
    "strengths": ["strength1", "strength2"],
    "recommendations": ["rec1", "rec2"],
    "approval_status": "approved|needs_revision",
    "confidence": 0.85
}}
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=review_prompt,
                system_message=self.system_prompt,
                metadata={"agent": self.name, "action": "review_code"}
            )
            
            # Try to parse review results
            try:
                review_data = json.loads(response.content)
                is_approved = review_data.get("approval_status") == "approved"
                
                return AgentObservation(
                    content=response.content,
                    success=is_approved,
                    metadata={
                        "action_type": "review_code",
                        "approval_status": review_data.get("approval_status"),
                        "overall_quality": review_data.get("overall_quality"),
                        "issues_count": len(review_data.get("issues_found", []))
                    }
                )
                
            except json.JSONDecodeError:
                return AgentObservation(
                    content=f"Code reviewed but not in valid JSON: {response.content}",
                    success=False,
                    metadata={"action_type": "review_code", "error": "invalid_json"}
                )
                
        except Exception as e:
            logger.error(f"Error reviewing code: {str(e)}")
            return AgentObservation(
                content=f"Error reviewing code: {str(e)}",
                success=False,
                metadata={"action_type": "review_code", "error": str(e)}
            )
    
    async def _fix_issues(self, task: str, issues: str) -> AgentObservation:
        """Fix identified issues in the code"""
        
        # Get the latest implementation
        implementation_obs = [
            obs for obs in self.session_memory.observations
            if obs.metadata.get("action_type") == "implement_code" and obs.success
        ]
        
        if not implementation_obs:
            return AgentObservation(
                content="No implementation found to fix",
                success=False,
                metadata={"action_type": "fix_issues", "error": "no_implementation"}
            )
        
        implementation = implementation_obs[-1].content
        
        fix_prompt = f"""
Coding task: {task}

Current implementation:
{implementation}

Issues to fix:
{issues}

Fix the identified issues and provide the corrected code. Ensure:
1. All identified issues are addressed
2. The fixes don't introduce new problems
3. The code still meets the original requirements
4. Proper testing is considered

Provide the fixed implementation in the same format as before.
"""
        
        try:
            response = await self.llm_engine.generate_async(
                prompt=fix_prompt,
                system_message=self.system_prompt,
                max_tokens=4000,
                metadata={"agent": self.name, "action": "fix_issues"}
            )
            
            # Store the fixed implementation
            await self.memory_manager.store_memory(
                content=response.content,
                metadata={
                    "agent": self.name,
                    "type": "implementation_fixed",
                    "task": task,
                    "original_issues": issues
                },
                category="code_implementations"
            )
            
            return AgentObservation(
                content=response.content,
                success=True,
                metadata={"action_type": "fix_issues"}
            )
            
        except Exception as e:
            logger.error(f"Error fixing issues: {str(e)}")
            return AgentObservation(
                content=f"Error fixing issues: {str(e)}",
                success=False,
                metadata={"action_type": "fix_issues", "error": str(e)}
            )
    
    def _extract_code_files(self, content: str) -> Dict[str, str]:
        """Extract code files from the implementation response"""
        
        files = {}
        
        # Pattern to match FILE: filename followed by code block
        file_pattern = r'FILE:\s*([^\n]+)\n```(?:python|javascript|typescript|go|rust|bash)?\n(.*?)\n```'
        
        matches = re.findall(file_pattern, content, re.DOTALL)
        
        for filename, code in matches:
            files[filename.strip()] = code.strip()
        
        return files