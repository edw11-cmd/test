"""
Core LLM Engine with OpenAI and Opik Integration
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from openai import OpenAI, AsyncOpenAI
import structlog
from pydantic import BaseModel

from .config import get_settings

# Try to import Opik, but handle gracefully if not available
try:
    from opik.integrations.openai import track_openai
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    def track_openai(client):
        """Fallback function when Opik is not available"""
        return client

logger = structlog.get_logger(__name__)
settings = get_settings()


class LLMResponse(BaseModel):
    """Structured LLM response"""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    metadata: Dict[str, Any] = {}


class LLMEngine:
    """Advanced LLM Engine with Opik tracking and optimization"""
    
    def __init__(self):
        self.settings = settings
        
        # Initialize OpenAI clients with Opik tracking
        self.sync_client = track_openai(OpenAI(api_key=self.settings.openai_api_key))
        self.async_client = track_openai(AsyncOpenAI(api_key=self.settings.openai_api_key))
        
        # Model configurations
        self.model_configs = {
            "gpt-4-turbo-preview": {
                "max_tokens": 4000,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            },
            "gpt-3.5-turbo": {
                "max_tokens": 2000,
                "temperature": 0.7,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0
            }
        }
        
        # Fallback models for error handling
        self.fallback_models = ["gpt-4-turbo-preview", "gpt-3.5-turbo"]
        
        if OPIK_AVAILABLE:
            logger.info("LLM Engine initialized with Opik tracking")
        else:
            logger.info("LLM Engine initialized (Opik not available - using fallback)")
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        return self.model_configs.get(model, self.model_configs["gpt-4-turbo-preview"])
    
    def _build_messages(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Build message array for OpenAI API"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": prompt})
        
        return messages
    
    async def generate_async(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate response asynchronously with error handling and fallbacks"""
        
        model = model or self.settings.openai_model
        config = self._get_model_config(model)
        
        # Override config with provided parameters
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        
        messages = self._build_messages(prompt, system_message, conversation_history)
        
        # Try primary model first, then fallbacks
        models_to_try = [model] + [m for m in self.fallback_models if m != model]
        
        for attempt_model in models_to_try:
            try:
                logger.info(f"Attempting generation with model: {attempt_model}")
                
                response = await self.async_client.chat.completions.create(
                    model=attempt_model,
                    messages=messages,
                    **config
                )
                
                result = LLMResponse(
                    content=response.choices[0].message.content,
                    model=attempt_model,
                    tokens_used=response.usage.total_tokens,
                    finish_reason=response.choices[0].finish_reason,
                    metadata=metadata or {}
                )
                
                logger.info(f"Successfully generated response with {attempt_model}")
                return result
                
            except Exception as e:
                logger.warning(f"Failed to generate with {attempt_model}: {str(e)}")
                if attempt_model == models_to_try[-1]:  # Last model
                    raise e
                continue
    
    def generate_sync(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Generate response synchronously"""
        
        model = model or self.settings.openai_model
        config = self._get_model_config(model)
        
        if temperature is not None:
            config["temperature"] = temperature
        if max_tokens is not None:
            config["max_tokens"] = max_tokens
        
        messages = self._build_messages(prompt, system_message, conversation_history)
        
        try:
            response = self.sync_client.chat.completions.create(
                model=model,
                messages=messages,
                **config
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata=metadata or {}
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            raise e
    
    async def generate_stream_async(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """Generate streaming response asynchronously"""
        
        model = model or self.settings.openai_model
        config = self._get_model_config(model)
        config.update(kwargs)
        
        messages = self._build_messages(prompt, system_message)
        
        try:
            stream = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                **config
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Failed to generate streaming response: {str(e)}")
            raise e


# Global LLM engine instance
llm_engine = LLMEngine()


def get_llm_engine() -> LLMEngine:
    """Get the global LLM engine instance"""
    return llm_engine