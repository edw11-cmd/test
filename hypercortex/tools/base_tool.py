"""
Base Tool Interface for HyperCortex-AI
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)


class ToolParameter(BaseModel):
    """Represents a tool parameter"""
    name: str
    type: str  # "string", "integer", "float", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


class ToolResult(BaseModel):
    """Represents the result of a tool execution"""
    success: bool
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}
    execution_time: float = 0.0
    timestamp: datetime = datetime.utcnow()


class BaseTool(ABC):
    """Base class for all tools in HyperCortex-AI"""
    
    def __init__(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        timeout: int = 30
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.timeout = timeout
        
        # Tool metadata
        self.usage_count = 0
        self.success_count = 0
        self.error_count = 0
        self.total_execution_time = 0.0
        
        logger.info(f"Initialized tool: {self.name}")
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        pass
    
    async def run(self, **kwargs) -> ToolResult:
        """Run the tool with validation and error handling"""
        
        start_time = datetime.utcnow()
        self.usage_count += 1
        
        try:
            # Validate parameters
            validation_result = self._validate_parameters(kwargs)
            if not validation_result["valid"]:
                self.error_count += 1
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Parameter validation failed: {validation_result['error']}",
                    metadata={"validation_error": True}
                )
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    self.execute(**kwargs),
                    timeout=self.timeout
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                result.execution_time = execution_time
                self.total_execution_time += execution_time
                
                if result.success:
                    self.success_count += 1
                else:
                    self.error_count += 1
                
                logger.info(f"Tool {self.name} executed successfully in {execution_time:.2f}s")
                return result
                
            except asyncio.TimeoutError:
                self.error_count += 1
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Tool execution timed out after {self.timeout}s",
                    metadata={"timeout_error": True}
                )
            
        except Exception as e:
            self.error_count += 1
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            logger.error(f"Error executing tool {self.name}: {str(e)}")
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                execution_time=execution_time,
                metadata={"unexpected_error": True}
            )
    
    def _validate_parameters(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool parameters"""
        
        # Check required parameters
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                return {
                    "valid": False,
                    "error": f"Required parameter '{param.name}' is missing"
                }
        
        # Check parameter types and values
        for param_name, value in kwargs.items():
            param = next((p for p in self.parameters if p.name == param_name), None)
            
            if param is None:
                return {
                    "valid": False,
                    "error": f"Unknown parameter '{param_name}'"
                }
            
            # Type validation
            if not self._validate_type(value, param.type):
                return {
                    "valid": False,
                    "error": f"Parameter '{param_name}' has invalid type. Expected {param.type}"
                }
            
            # Enum validation
            if param.enum and value not in param.enum:
                return {
                    "valid": False,
                    "error": f"Parameter '{param_name}' value must be one of {param.enum}"
                }
        
        return {"valid": True}
    
    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """Validate parameter type"""
        
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "float":
            return isinstance(value, (int, float))
        elif expected_type == "boolean":
            return isinstance(value, bool)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        else:
            return True  # Unknown type, allow it
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool schema for documentation/API"""
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [param.dict() for param in self.parameters],
            "timeout": self.timeout
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        
        success_rate = (self.success_count / self.usage_count) if self.usage_count > 0 else 0
        avg_execution_time = (self.total_execution_time / self.usage_count) if self.usage_count > 0 else 0
        
        return {
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "total_execution_time": self.total_execution_time,
            "average_execution_time": avg_execution_time
        }


class ToolRegistry:
    """Registry for managing tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        logger.info("Tool Registry initialized")
    
    def register_tool(self, tool: BaseTool):
        """Register a tool"""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool"""
        if tool_name in self.tools:
            del self.tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tools"""
        return list(self.tools.keys())
    
    def get_tool_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Get schemas for all tools"""
        return {name: tool.get_schema() for name, tool in self.tools.items()}
    
    def get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tools"""
        return {name: tool.get_stats() for name, tool in self.tools.items()}
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tool '{tool_name}' not found",
                metadata={"tool_not_found": True}
            )
        
        return await tool.run(**kwargs)


# Global tool registry
tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry"""
    return tool_registry