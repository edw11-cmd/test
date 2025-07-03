"""
Built-in Tools for HyperCortex-AI
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional
import httpx
import structlog

from .base_tool import BaseTool, ToolParameter, ToolResult

logger = structlog.get_logger(__name__)


class WebSearchTool(BaseTool):
    """Tool for web search (simulated)"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results",
                required=False,
                default=5
            )
        ]
        
        super().__init__(
            name="web_search",
            description="Search the web for information",
            parameters=parameters,
            timeout=30
        )
    
    async def execute(self, query: str, max_results: int = 5) -> ToolResult:
        """Execute web search (simulated)"""
        
        # Simulate web search results
        simulated_results = [
            {
                "title": f"Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a simulated search result snippet for query '{query}'. It contains relevant information about the topic.",
                "relevance_score": 0.9 - (i * 0.1)
            }
            for i in range(min(max_results, 5))
        ]
        
        return ToolResult(
            success=True,
            result={
                "query": query,
                "results": simulated_results,
                "total_results": len(simulated_results)
            },
            metadata={
                "simulated": True,
                "search_engine": "simulated"
            }
        )


class CodeExecutorTool(BaseTool):
    """Tool for executing code safely"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="code",
                type="string",
                description="Code to execute",
                required=True
            ),
            ToolParameter(
                name="language",
                type="string",
                description="Programming language",
                required=False,
                default="python",
                enum=["python", "javascript", "bash"]
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Execution timeout in seconds",
                required=False,
                default=10
            )
        ]
        
        super().__init__(
            name="code_executor",
            description="Execute code in a safe environment",
            parameters=parameters,
            timeout=30
        )
    
    async def execute(self, code: str, language: str = "python", timeout: int = 10) -> ToolResult:
        """Execute code safely"""
        
        try:
            if language == "python":
                return await self._execute_python(code, timeout)
            elif language == "javascript":
                return await self._execute_javascript(code, timeout)
            elif language == "bash":
                return await self._execute_bash(code, timeout)
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Unsupported language: {language}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Code execution error: {str(e)}"
            )
    
    async def _execute_python(self, code: str, timeout: int) -> ToolResult:
        """Execute Python code"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return ToolResult(
                success=result.returncode == 0,
                result={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                },
                error=result.stderr if result.returncode != 0 else None,
                metadata={"language": "python"}
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                result=None,
                error=f"Code execution timed out after {timeout}s"
            )
        finally:
            os.unlink(temp_file)
    
    async def _execute_javascript(self, code: str, timeout: int) -> ToolResult:
        """Execute JavaScript code"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ["node", temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return ToolResult(
                success=result.returncode == 0,
                result={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                },
                error=result.stderr if result.returncode != 0 else None,
                metadata={"language": "javascript"}
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                result=None,
                error=f"Code execution timed out after {timeout}s"
            )
        except FileNotFoundError:
            return ToolResult(
                success=False,
                result=None,
                error="Node.js not found. Please install Node.js to execute JavaScript code."
            )
        finally:
            os.unlink(temp_file)
    
    async def _execute_bash(self, code: str, timeout: int) -> ToolResult:
        """Execute Bash code"""
        
        try:
            result = subprocess.run(
                code,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return ToolResult(
                success=result.returncode == 0,
                result={
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "return_code": result.returncode
                },
                error=result.stderr if result.returncode != 0 else None,
                metadata={"language": "bash"}
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                result=None,
                error=f"Code execution timed out after {timeout}s"
            )


class FileManagerTool(BaseTool):
    """Tool for file operations"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="operation",
                type="string",
                description="File operation to perform",
                required=True,
                enum=["read", "write", "list", "delete", "exists"]
            ),
            ToolParameter(
                name="path",
                type="string",
                description="File or directory path",
                required=True
            ),
            ToolParameter(
                name="content",
                type="string",
                description="Content to write (for write operation)",
                required=False
            ),
            ToolParameter(
                name="encoding",
                type="string",
                description="File encoding",
                required=False,
                default="utf-8"
            )
        ]
        
        super().__init__(
            name="file_manager",
            description="Perform file and directory operations",
            parameters=parameters,
            timeout=15
        )
    
    async def execute(
        self,
        operation: str,
        path: str,
        content: str = None,
        encoding: str = "utf-8"
    ) -> ToolResult:
        """Execute file operation"""
        
        try:
            if operation == "read":
                return await self._read_file(path, encoding)
            elif operation == "write":
                return await self._write_file(path, content, encoding)
            elif operation == "list":
                return await self._list_directory(path)
            elif operation == "delete":
                return await self._delete_file(path)
            elif operation == "exists":
                return await self._check_exists(path)
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"File operation error: {str(e)}"
            )
    
    async def _read_file(self, path: str, encoding: str) -> ToolResult:
        """Read file content"""
        
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                result=None,
                error=f"File not found: {path}"
            )
        
        if not os.path.isfile(path):
            return ToolResult(
                success=False,
                result=None,
                error=f"Path is not a file: {path}"
            )
        
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                result={
                    "content": content,
                    "size": len(content),
                    "encoding": encoding
                },
                metadata={"operation": "read", "path": path}
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                success=False,
                result=None,
                error=f"Cannot decode file with encoding {encoding}"
            )
    
    async def _write_file(self, path: str, content: str, encoding: str) -> ToolResult:
        """Write content to file"""
        
        if content is None:
            return ToolResult(
                success=False,
                result=None,
                error="Content is required for write operation"
            )
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                result={
                    "bytes_written": len(content.encode(encoding)),
                    "encoding": encoding
                },
                metadata={"operation": "write", "path": path}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Write error: {str(e)}"
            )
    
    async def _list_directory(self, path: str) -> ToolResult:
        """List directory contents"""
        
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                result=None,
                error=f"Directory not found: {path}"
            )
        
        if not os.path.isdir(path):
            return ToolResult(
                success=False,
                result=None,
                error=f"Path is not a directory: {path}"
            )
        
        try:
            items = []
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                items.append({
                    "name": item,
                    "type": "directory" if os.path.isdir(item_path) else "file",
                    "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                })
            
            return ToolResult(
                success=True,
                result={
                    "items": items,
                    "count": len(items)
                },
                metadata={"operation": "list", "path": path}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"List error: {str(e)}"
            )
    
    async def _delete_file(self, path: str) -> ToolResult:
        """Delete file or directory"""
        
        if not os.path.exists(path):
            return ToolResult(
                success=False,
                result=None,
                error=f"Path not found: {path}"
            )
        
        try:
            if os.path.isfile(path):
                os.remove(path)
                return ToolResult(
                    success=True,
                    result={"deleted": "file"},
                    metadata={"operation": "delete", "path": path}
                )
            elif os.path.isdir(path):
                os.rmdir(path)  # Only removes empty directories
                return ToolResult(
                    success=True,
                    result={"deleted": "directory"},
                    metadata={"operation": "delete", "path": path}
                )
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Unknown path type: {path}"
                )
                
        except OSError as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Delete error: {str(e)}"
            )
    
    async def _check_exists(self, path: str) -> ToolResult:
        """Check if path exists"""
        
        exists = os.path.exists(path)
        
        result = {"exists": exists}
        
        if exists:
            result.update({
                "type": "directory" if os.path.isdir(path) else "file",
                "size": os.path.getsize(path) if os.path.isfile(path) else None
            })
        
        return ToolResult(
            success=True,
            result=result,
            metadata={"operation": "exists", "path": path}
        )


class HttpClientTool(BaseTool):
    """Tool for HTTP requests"""
    
    def __init__(self):
        parameters = [
            ToolParameter(
                name="method",
                type="string",
                description="HTTP method",
                required=True,
                enum=["GET", "POST", "PUT", "DELETE", "PATCH"]
            ),
            ToolParameter(
                name="url",
                type="string",
                description="Request URL",
                required=True
            ),
            ToolParameter(
                name="headers",
                type="object",
                description="Request headers",
                required=False
            ),
            ToolParameter(
                name="data",
                type="object",
                description="Request data/body",
                required=False
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Request timeout in seconds",
                required=False,
                default=10
            )
        ]
        
        super().__init__(
            name="http_client",
            description="Make HTTP requests",
            parameters=parameters,
            timeout=30
        )
    
    async def execute(
        self,
        method: str,
        url: str,
        headers: Dict[str, str] = None,
        data: Any = None,
        timeout: int = 10
    ) -> ToolResult:
        """Execute HTTP request"""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data if isinstance(data, dict) else None,
                    data=data if not isinstance(data, dict) else None,
                    timeout=timeout
                )
                
                # Try to parse JSON response
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                return ToolResult(
                    success=response.is_success,
                    result={
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "data": response_data,
                        "url": str(response.url)
                    },
                    error=None if response.is_success else f"HTTP {response.status_code}: {response.text}",
                    metadata={
                        "method": method,
                        "status_code": response.status_code
                    }
                )
                
        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                result=None,
                error=f"Request timed out after {timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"HTTP request error: {str(e)}"
            )


def register_builtin_tools():
    """Register all built-in tools"""
    
    from .base_tool import get_tool_registry
    
    registry = get_tool_registry()
    
    # Register tools
    registry.register_tool(WebSearchTool())
    registry.register_tool(CodeExecutorTool())
    registry.register_tool(FileManagerTool())
    registry.register_tool(HttpClientTool())
    
    logger.info("Registered built-in tools")