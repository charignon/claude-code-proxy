from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Union, Literal
import httpx
import os
from fastapi.responses import JSONResponse, StreamingResponse
import litellm
import uuid
import time
from dotenv import load_dotenv
import re
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.WARN,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
import uvicorn
# Tell uvicorn's loggers to be quiet
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# Create a filter to block any log messages containing specific strings
class MessageFilter(logging.Filter):
    def filter(self, record):
        # Block messages containing these strings
        blocked_phrases = [
            "LiteLLM completion()",
            "HTTP Request:", 
            "selected model name for cost calculation",
            "utils.py",
            "cost_calculator"
        ]
        
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for phrase in blocked_phrases:
                if phrase in record.msg:
                    return False
        return True

# Apply the filter to the root logger to catch all messages
root_logger = logging.getLogger()
root_logger.addFilter(MessageFilter())

# Custom formatter for model mapping logs
class ColorizedFormatter(logging.Formatter):
    """Custom formatter to highlight model mappings"""
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def format(self, record):
        if record.levelno == logging.debug and "MODEL MAPPING" in record.msg:
            # Apply colors and formatting to model mapping logs
            return f"{self.BOLD}{self.GREEN}{record.msg}{self.RESET}"
        return super().format(record)

# Apply custom formatter to console handler
for handler in logger.handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setFormatter(ColorizedFormatter('%(asctime)s - %(levelname)s - %(message)s'))

app = FastAPI()

# Get API keys from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Get Vertex AI project and location from environment (if set)
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "unset")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "unset")

# Option to use Gemini API key instead of ADC for Vertex AI
USE_VERTEX_AUTH = os.environ.get("USE_VERTEX_AUTH", "False").lower() == "true"

# Get OpenAI base URL from environment (if set)
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Log startup config for debugging
if OPENAI_BASE_URL and "llm-proxy" in OPENAI_BASE_URL:
    print(f"🔀 LLM_PROXY mode ENABLED: {OPENAI_BASE_URL}", file=sys.stderr)
else:
    print(f"📡 Direct OpenAI mode (OPENAI_BASE_URL={OPENAI_BASE_URL})", file=sys.stderr)

# Get preferred provider (default to openai)
PREFERRED_PROVIDER = os.environ.get("PREFERRED_PROVIDER", "openai").lower()

# Get model mapping configuration from environment
# Default to latest OpenAI models if not set
BIG_MODEL = os.environ.get("BIG_MODEL", "gpt-4.1")
SMALL_MODEL = os.environ.get("SMALL_MODEL", "gpt-4.1-mini")

# Fix for OpenAI models that don't handle empty tool results well
# When enabled, replaces empty content with a placeholder
FIX_EMPTY_TOOL_RESULTS = os.environ.get("FIX_EMPTY_TOOL_RESULTS", "").lower() in ("1", "true", "yes")
EMPTY_TOOL_RESULT_PLACEHOLDER = os.environ.get("EMPTY_TOOL_RESULT_PLACEHOLDER", "NULL")

# List of OpenAI models
# Naming convention:
#   gpt-4o      = GPT-4 "omni" - multimodal (text, image, audio)
#   gpt-4.1     = April 2025 release, better coding & instruction following, 1M context
#   gpt-5       = August 2025 flagship, reasoning model with adjustable effort
#   -mini       = Smaller, faster, cheaper variant
#   -nano       = Smallest, fastest, cheapest variant
#   o1/o3       = Reasoning-first models (chain of thought)
OPENAI_MODELS = [
    # GPT-5 family (Aug 2025) - reasoning models
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    # GPT-4.1 family (Apr 2025) - fast, 1M context
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    # GPT-4o family - multimodal
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4o-audio-preview",
    "gpt-4o-mini-audio-preview",
    "chatgpt-4o-latest",
    # Reasoning models
    "o3",
    "o3-mini",
    "o1",
    "o1-mini",
    "o1-pro",
    # Legacy
    "gpt-4.5-preview",
]

# List of Gemini models
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro"
]

# ============================================================================
# Cost Tracking for Hammerspoon Widget
# ============================================================================
STATS_FILE = "/tmp/claude-code-proxy-stats.json"

# Pricing per 1M tokens (input, output) - Dec 2024 prices
MODEL_PRICING = {
    # GPT-5.x family
    "gpt-5.2": (2.50, 10.00),
    "gpt-5.1": (2.50, 10.00),
    "gpt-5": (2.50, 10.00),
    "gpt-5-mini": (0.30, 1.20),
    "gpt-5-nano": (0.10, 0.40),
    # GPT-4.1 family
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    # GPT-4o family
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Reasoning models
    "o3": (10.00, 40.00),
    "o3-mini": (1.10, 4.40),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    # Gemini
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-2.5-pro": (1.25, 5.00),
    # Default fallback
    "default": (2.00, 8.00),
}

# Claude pricing per 1M tokens (input, output)
CLAUDE_PRICING = {
    "claude-sonnet": (3.00, 15.00),  # Claude 4 Sonnet
    "claude-haiku": (0.80, 4.00),    # Claude 4 Haiku
    "claude-opus": (15.00, 75.00),   # Claude 4 Opus
}

# Map OpenAI models to equivalent Claude tier for comparison
MODEL_TO_CLAUDE_TIER = {
    # Big models -> Sonnet
    "gpt-5.2": "claude-sonnet",
    "gpt-5.1": "claude-sonnet",
    "gpt-5": "claude-sonnet",
    "gpt-4.1": "claude-sonnet",
    "gpt-4o": "claude-sonnet",
    # Mini models -> Haiku
    "gpt-5-mini": "claude-haiku",
    "gpt-4.1-mini": "claude-haiku",
    "gpt-4o-mini": "claude-haiku",
    "gpt-5-nano": "claude-haiku",
    "gpt-4.1-nano": "claude-haiku",
    # Reasoning -> Opus (expensive)
    "o3": "claude-opus",
    "o3-mini": "claude-sonnet",
    "o1": "claude-opus",
    "o1-mini": "claude-sonnet",
    # Gemini
    "gemini-2.5-flash": "claude-haiku",
    "gemini-2.5-pro": "claude-sonnet",
    # Default
    "default": "claude-sonnet",
}

# In-memory stats (persisted to file, rotates daily at midnight)
def _load_or_init_stats():
    """Load existing stats if from today, otherwise start fresh."""
    today = datetime.now().date().isoformat()
    try:
        with open(STATS_FILE, "r") as f:
            existing = json.load(f)
            # Check if stats are from today
            if "session_start" in existing:
                session_date = existing["session_start"][:10]  # Extract YYYY-MM-DD
                if session_date == today:
                    logger.info(f"Loaded existing stats from today ({existing.get('totals', {}).get('requests', 0)} requests)")
                    return existing
            logger.info("Stats from previous day, starting fresh")
    except (FileNotFoundError, json.JSONDecodeError):
        logger.info("No existing stats file, starting fresh")

    return {
        "models": {},
        "totals": {"input_tokens": 0, "output_tokens": 0, "total_cost": 0, "claude_cost": 0, "requests": 0},
        "session_start": datetime.now().isoformat(),
    }

_usage_stats = _load_or_init_stats()

def _get_model_pricing(model_name: str) -> tuple:
    """Get (input_price, output_price) per 1M tokens for a model."""
    # Strip provider prefix
    clean_name = model_name
    for prefix in ["openai/", "gemini/", "anthropic/"]:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break

    # Try exact match first
    if clean_name in MODEL_PRICING:
        return MODEL_PRICING[clean_name]

    # Try partial match
    for model, pricing in MODEL_PRICING.items():
        if model in clean_name or clean_name in model:
            return pricing

    return MODEL_PRICING["default"]

def estimate_tokens(text: str) -> int:
    """Estimate token count from text (~4 chars per token for English)."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def _get_claude_equivalent_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate what the cost would be with equivalent Claude model."""
    # Strip provider prefix
    clean_name = model
    for prefix in ["openai/", "gemini/", "anthropic/"]:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            break

    # Find Claude tier for this model
    claude_tier = MODEL_TO_CLAUDE_TIER.get(clean_name)
    if not claude_tier:
        # Try partial match
        for model_pattern, tier in MODEL_TO_CLAUDE_TIER.items():
            if model_pattern in clean_name or clean_name in model_pattern:
                claude_tier = tier
                break
    if not claude_tier:
        claude_tier = "claude-sonnet"  # default

    input_price, output_price = CLAUDE_PRICING[claude_tier]
    return (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)

def track_usage(model: str, input_tokens: int, output_tokens: int, input_text: str = "", output_text: str = ""):
    """Track usage and update stats file for Hammerspoon widget."""
    global _usage_stats

    # Debug log
    print(f"[COST] track_usage called: model={model}, in={input_tokens}, out={output_tokens}, in_text_len={len(input_text)}, out_text_len={len(output_text)}", flush=True)

    # Estimate tokens from text if not provided
    if input_tokens == 0 and input_text:
        input_tokens = estimate_tokens(input_text)
    if output_tokens == 0 and output_text:
        output_tokens = estimate_tokens(output_text)

    # Get pricing for actual model
    input_price, output_price = _get_model_pricing(model)
    cost = (input_tokens * input_price / 1_000_000) + (output_tokens * output_price / 1_000_000)

    # Get equivalent Claude cost
    claude_cost = _get_claude_equivalent_cost(model, input_tokens, output_tokens)

    # Strip provider prefix for display
    display_model = model
    for prefix in ["openai/", "gemini/", "anthropic/"]:
        if display_model.startswith(prefix):
            display_model = display_model[len(prefix):]
            break

    # Update model stats
    if display_model not in _usage_stats["models"]:
        _usage_stats["models"][display_model] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0,
            "claude_cost": 0,
            "requests": 0,
        }

    _usage_stats["models"][display_model]["input_tokens"] += input_tokens
    _usage_stats["models"][display_model]["output_tokens"] += output_tokens
    _usage_stats["models"][display_model]["total_cost"] += cost
    _usage_stats["models"][display_model]["claude_cost"] += claude_cost
    _usage_stats["models"][display_model]["requests"] += 1

    # Update totals
    _usage_stats["totals"]["input_tokens"] += input_tokens
    _usage_stats["totals"]["output_tokens"] += output_tokens
    _usage_stats["totals"]["total_cost"] += cost
    _usage_stats["totals"]["claude_cost"] += claude_cost
    _usage_stats["totals"]["requests"] += 1

    # Write to file
    try:
        with open(STATS_FILE, "w") as f:
            json.dump(_usage_stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write stats file: {e}")

# Helper function to clean schema for Gemini
def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()): # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema


def has_web_search_tool(tools: Optional[List[Any]]) -> bool:
    """Check if the tools list contains a web_search tool (Anthropic-defined tool)."""
    if not tools:
        return False
    for tool in tools:
        tool_dict = tool.dict() if hasattr(tool, 'dict') else tool
        tool_type = tool_dict.get("type", "")
        if tool_type and tool_type.startswith("web_search"):
            return True
    return False


def convert_anthropic_messages_to_input(messages: List[Any], system: Optional[Any] = None) -> str:
    """Convert Anthropic messages format to a single input string for OpenAI Responses API."""
    parts = []

    # Add system message if present
    if system:
        if isinstance(system, str):
            parts.append(f"System: {system}")
        elif isinstance(system, list):
            system_text = ""
            for block in system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n"
            if system_text:
                parts.append(f"System: {system_text.strip()}")

    # Convert messages
    for msg in messages:
        role = msg.role if hasattr(msg, 'role') else msg.get('role', 'user')
        content = msg.content if hasattr(msg, 'content') else msg.get('content', '')

        if isinstance(content, str):
            parts.append(f"{role.capitalize()}: {content}")
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if hasattr(block, 'type'):
                    if block.type == "text":
                        text_parts.append(block.text)
                    elif block.type == "tool_result":
                        result_content = parse_tool_result_content(block.content if hasattr(block, 'content') else "")
                        text_parts.append(f"[Tool Result: {result_content}]")
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        result_content = parse_tool_result_content(block.get("content", ""))
                        text_parts.append(f"[Tool Result: {result_content}]")
            if text_parts:
                parts.append(f"{role.capitalize()}: {' '.join(text_parts)}")

    return "\n\n".join(parts)


async def call_openai_responses_api_non_streaming(
    model: str,
    input_text: str,
    api_key: str = None
) -> Dict[str, Any]:
    """Call OpenAI's Responses API for web search (non-streaming)."""
    openai_model = get_openai_web_search_model(model)

    # Use llm-proxy if configured, otherwise direct OpenAI
    if OPENAI_BASE_URL and "llm-proxy" in OPENAI_BASE_URL:
        base_url = OPENAI_BASE_URL.rstrip("/")
        url = f"{base_url}/responses"
    else:
        url = "https://api.openai.com/v1/responses"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Add usecase header for llm-proxy
    if OPENAI_BASE_URL and "llm-proxy" in OPENAI_BASE_URL:
        headers["X-Usecase"] = "claude-hacky"

    payload = {
        "model": openai_model,
        "input": input_text,
        "tools": [{"type": "web_search"}]
    }

    logger.debug(f"Calling OpenAI Responses API with model={openai_model}, stream=False")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {response.text}")
        return response.json()


def get_openai_web_search_model(model: str) -> str:
    """Map model name to OpenAI model that supports web search.

    Supported models: gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano,
                      gpt-5, gpt-5.1, gpt-5.2, o3
    """
    openai_model = model
    if model.startswith("openai/"):
        openai_model = model[7:]  # Remove "openai/" prefix

    # Map to appropriate model for web search in Responses API
    lower_model = openai_model.lower()

    # gpt-5.x series - use as-is, they support web search
    if "gpt-5.2" in lower_model:
        return "gpt-5.2"
    elif "gpt-5.1" in lower_model:
        return "gpt-5.1"
    elif "gpt-5" in lower_model:
        return "gpt-5"
    # gpt-4.1 series
    elif "gpt-4.1-nano" in lower_model:
        return "gpt-4.1-nano"
    elif "gpt-4.1-mini" in lower_model:
        return "gpt-4.1-mini"
    elif "gpt-4.1" in lower_model:
        return "gpt-4.1"
    # Mini models
    elif "mini" in lower_model:
        return "gpt-4.1-mini"
    # Default to gpt-4.1 for web search (better than gpt-4o)
    else:
        return "gpt-4.1"


def convert_openai_responses_to_anthropic(
    openai_response: Dict[str, Any],
    original_request: Any
) -> Dict[str, Any]:
    """Convert OpenAI Responses API response to Anthropic message format."""
    content = []

    # Extract output from OpenAI response
    output_items = openai_response.get("output", [])

    # Track if we've added web search results
    search_results_added = False

    for item in output_items:
        item_type = item.get("type", "")

        if item_type == "web_search_call":
            # Add server_tool_use block for the search
            tool_id = item.get("id", f"srvtoolu_{uuid.uuid4().hex[:24]}")
            status = item.get("status", "completed")

            # Extract search query if available
            search_input = {}
            if "action" in item:
                action = item.get("action", {})
                if isinstance(action, dict) and action.get("type") == "search":
                    search_input["query"] = action.get("query", "")

            content.append({
                "type": "server_tool_use",
                "id": tool_id,
                "name": "web_search",
                "input": search_input
            })

            # Add a placeholder web_search_tool_result
            if not search_results_added:
                content.append({
                    "type": "web_search_tool_result",
                    "tool_use_id": tool_id,
                    "content": [{
                        "type": "web_search_result",
                        "url": "https://search.openai.com",
                        "title": "Web Search Results",
                        "encrypted_content": "web_search_via_openai"
                    }]
                })
                search_results_added = True

        elif item_type == "message" or item_type == "output_text":
            # Handle text output with potential citations
            text = item.get("text", "") or item.get("content", "")
            if isinstance(text, list):
                # Sometimes content is a list of content blocks
                for text_block in text:
                    if isinstance(text_block, dict) and text_block.get("type") == "output_text":
                        text = text_block.get("text", "")
                        break
                    elif isinstance(text_block, str):
                        text = text_block
                        break

            if not text:
                continue

            annotations = item.get("annotations", [])

            if annotations:
                # Convert OpenAI annotations to Anthropic citations
                citations = []
                for ann in annotations:
                    if ann.get("type") == "url_citation":
                        citations.append({
                            "type": "web_search_result_location",
                            "url": ann.get("url", ""),
                            "title": ann.get("title", ""),
                            "encrypted_index": f"idx_{uuid.uuid4().hex[:16]}",
                            "cited_text": text[ann.get("start_index", 0):ann.get("end_index", len(text))][:150]
                        })

                content.append({
                    "type": "text",
                    "text": text,
                    "citations": citations if citations else None
                })
            else:
                content.append({
                    "type": "text",
                    "text": text
                })

    # If no content was extracted, add a fallback
    if not content:
        # Try to extract text from the response directly
        if "output_text" in openai_response:
            content.append({
                "type": "text",
                "text": openai_response.get("output_text", "")
            })
        else:
            content.append({
                "type": "text",
                "text": "Web search completed but no results were returned."
            })

    # Build the Anthropic response
    response_id = openai_response.get("id", f"msg_{uuid.uuid4().hex[:24]}")

    # Get usage info
    usage_info = openai_response.get("usage", {})
    input_tokens = usage_info.get("input_tokens", 0) or usage_info.get("prompt_tokens", 0)
    output_tokens = usage_info.get("output_tokens", 0) or usage_info.get("completion_tokens", 0)

    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": original_request.model,
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "server_tool_use": {
                "web_search_requests": 1
            }
        }
    }


async def handle_web_search_streaming(
    model: str,
    input_text: str,
    api_key: str,
    original_request: Any
):
    """Handle streaming responses from OpenAI Responses API and convert to Anthropic format."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send message_start event
    message_data = {
        'type': 'message_start',
        'message': {
            'id': message_id,
            'type': 'message',
            'role': 'assistant',
            'model': original_request.model,
            'content': [],
            'stop_reason': None,
            'stop_sequence': None,
            'usage': {
                'input_tokens': 0,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'output_tokens': 0,
                'server_tool_use': {
                    'web_search_requests': 0
                }
            }
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"

    # Send content_block_start for text
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    # Send ping
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    accumulated_text = ""
    output_tokens = 0

    openai_model = get_openai_web_search_model(model)

    # Use llm-proxy if configured, otherwise direct OpenAI
    if OPENAI_BASE_URL and "llm-proxy" in OPENAI_BASE_URL:
        base_url = OPENAI_BASE_URL.rstrip("/")
        url = f"{base_url}/responses"
    else:
        url = "https://api.openai.com/v1/responses"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Add usecase header for llm-proxy
    if OPENAI_BASE_URL and "llm-proxy" in OPENAI_BASE_URL:
        headers["X-Usecase"] = "claude-hacky"

    payload = {
        "model": openai_model,
        "input": input_text,
        "tools": [{"type": "web_search"}],
        "stream": True
    }

    logger.debug(f"Calling OpenAI Responses API with model={openai_model}, stream=True, url={url}")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_msg = f"OpenAI API error: {error_text.decode()}"
                    logger.error(error_msg)
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': error_msg}})}\n\n"
                else:
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue

                        try:
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                break

                            data = json.loads(data_str)

                            # Handle different event types from OpenAI Responses API
                            event_type = data.get("type", "")

                            if event_type == "response.output_text.delta":
                                delta_text = data.get("delta", "")
                                if delta_text:
                                    accumulated_text += delta_text
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_text}})}\n\n"

                            elif event_type == "response.done":
                                # Extract usage info
                                response_data = data.get("response", {})
                                usage = response_data.get("usage", {})
                                output_tokens = usage.get("output_tokens", 0)

                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.error(f"Error processing streaming chunk: {e}")
                            continue

    except Exception as e:
        logger.error(f"Error in web search streaming: {e}")
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': f'Error during web search: {str(e)}'}})}\n\n"

    # Close text block
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # Send message_delta with stop reason and web_search usage
    usage_data = {
        'output_tokens': output_tokens,
        'server_tool_use': {
            'web_search_requests': 1
        }
    }
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage_data})}\n\n"

    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    # Send final [DONE]
    yield "data: [DONE]\n\n"


async def stream_direct_to_llm_proxy(litellm_request: dict, original_request, base_url: str, api_key: str):
    """Stream directly to llm-proxy, bypassing LiteLLM (which strips tools with custom api_base)."""
    import httpx

    litellm_tools = litellm_request.get("tools", [])
    original_tools = original_request.tools if original_request.tools else []

    # Check if this is Ollama (studio.lan) - disable streaming for tool calls
    # Ollama has a bug where streaming tool calls are malformed (missing index field)
    is_ollama = "11434" in base_url or "ollama" in base_url.lower()
    has_tools = bool(litellm_tools or original_tools)
    use_streaming = not (is_ollama and has_tools)

    if is_ollama and has_tools:
        logger.info("🔧 Ollama + tools detected: using non-streaming mode for reliable tool calls")

    # Build OpenAI-compatible request
    openai_request = {
        "model": litellm_request.get("model", "").replace("openai/", ""),
        "messages": litellm_request.get("messages", []),
        "stream": use_streaming,
    }

    if use_streaming:
        openai_request["stream_options"] = {"include_usage": True}

    # Use tools from litellm_request or convert from original request
    tools_to_use = litellm_tools if litellm_tools else None
    if not tools_to_use and original_tools:
        # Convert Anthropic tools to OpenAI format
        tools_to_use = []
        for tool in original_tools:
            tool_dict = tool.dict() if hasattr(tool, 'dict') else tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict.get("name", ""),
                    "description": tool_dict.get("description", ""),
                    "parameters": tool_dict.get("input_schema", {})
                }
            }
            tools_to_use.append(openai_tool)

    if tools_to_use:
        openai_request["tools"] = tools_to_use
    if litellm_request.get("tool_choice"):
        openai_request["tool_choice"] = litellm_request["tool_choice"]
    if litellm_request.get("max_completion_tokens"):
        openai_request["max_completion_tokens"] = litellm_request["max_completion_tokens"]
    elif litellm_request.get("max_tokens"):
        openai_request["max_tokens"] = litellm_request["max_tokens"]
    if litellm_request.get("temperature"):
        openai_request["temperature"] = litellm_request["temperature"]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Usecase": "claude-hacky",
    }

    url = f"{base_url.rstrip('/')}/chat/completions"

    # For non-streaming mode (Ollama + tools), make a single request and convert to SSE
    if not use_streaming:
        async def generate_non_streaming():
            message_id = f"msg_{uuid.uuid4().hex[:24]}"

            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(url, json=openai_request, headers=headers)
                if response.status_code != 200:
                    raise HTTPException(status_code=response.status_code, detail=f"Ollama error: {response.text}")

                result = response.json()

            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason", "end_turn")
            usage = result.get("usage", {})

            # Map finish reasons
            if finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason == "stop":
                stop_reason = "end_turn"
            else:
                stop_reason = finish_reason

            # Build content blocks
            content_blocks = []
            if content:
                content_blocks.append({"type": "text", "text": content})

            for tc in tool_calls:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except:
                    args = {}
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                    "name": func.get("name", ""),
                    "input": args
                })

            # Emit SSE events (Claude Code expects streaming format even for non-streaming)
            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'model': original_request.model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0, 'output_tokens': 0}}})}\n\n"

            # Emit content blocks
            for idx, block in enumerate(content_blocks):
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': block})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

            # Emit message_delta with stop_reason and usage
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': usage.get('completion_tokens', 0)}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_non_streaming(), media_type="text/event-stream")

    async def generate():
        message_id = f"msg_{uuid.uuid4().hex[:24]}"

        # Send message_start event (required by Anthropic format)
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"

        # Send content_block_start for text (index 0)
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

        # Send ping
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

        # Track state for proper event emission
        text_block_closed = False
        tool_calls_by_index = {}  # Track tool calls by their OpenAI index
        anthropic_tool_index = 0  # Anthropic index for tool blocks (starts at 1, after text block at 0)
        openai_to_anthropic_index = {}  # Map OpenAI tool index to Anthropic block index
        input_tokens = 0
        output_tokens = 0
        finished = False
        final_stop_reason = "end_turn"
        accumulated_output = []  # Track output text for token estimation

        # Calculate input text for estimation
        input_text = json.dumps(openai_request.get("messages", []))

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", url, json=openai_request, headers=headers) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=f"llm-proxy error: {error_text.decode()}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)

                            # Extract usage if present (chunk["usage"] can be None)
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                                input_tokens = usage.get("prompt_tokens", input_tokens)
                                output_tokens = usage.get("completion_tokens", output_tokens)

                            if chunk.get("choices") and len(chunk["choices"]) > 0:
                                choice = chunk["choices"][0]
                                delta = choice.get("delta", {})
                                finish_reason = choice.get("finish_reason")

                                # Handle content delta
                                if "content" in delta and delta["content"] and not text_block_closed:
                                    accumulated_output.append(delta["content"])
                                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"

                                # Handle tool calls
                                if "tool_calls" in delta:
                                    for tc in delta["tool_calls"]:
                                        openai_index = tc.get("index", 0)

                                        # Check if this is a new tool call (has id and name)
                                        if tc.get("id") or tc.get("function", {}).get("name"):
                                            # Close text block if not already closed
                                            if not text_block_closed:
                                                text_block_closed = True
                                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                                            # New tool call - emit content_block_start
                                            if openai_index not in openai_to_anthropic_index:
                                                anthropic_tool_index += 1
                                                openai_to_anthropic_index[openai_index] = anthropic_tool_index

                                                tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                                                tool_name = tc.get("function", {}).get("name", "")

                                                tool_calls_by_index[openai_index] = {
                                                    "id": tool_id,
                                                    "name": tool_name,
                                                    "anthropic_index": anthropic_tool_index
                                                }

                                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': {}}})}\n\n"

                                        # Handle arguments delta
                                        if tc.get("function", {}).get("arguments"):
                                            # Make sure we have the anthropic index for this tool
                                            if openai_index in openai_to_anthropic_index:
                                                ant_idx = openai_to_anthropic_index[openai_index]
                                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': ant_idx, 'delta': {'type': 'input_json_delta', 'partial_json': tc['function']['arguments']}})}\n\n"

                                # Handle finish_reason - don't return yet, wait for usage chunk
                                if finish_reason and not finished:
                                    finished = True
                                    final_stop_reason = "end_turn"
                                    if finish_reason == "length":
                                        final_stop_reason = "max_tokens"
                                        # Inject warning into response so user sees it (index 0 is always text block)
                                        warning_text = "\n\n⚠️ **Hit max_tokens limit** - response was cut off. Say 'continue' to proceed."
                                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': warning_text}})}\n\n"
                                    elif finish_reason == "tool_calls":
                                        final_stop_reason = "tool_use"
                                    elif finish_reason == "stop":
                                        final_stop_reason = "end_turn"

                                    # Close text block if not already closed
                                    if not text_block_closed:
                                        text_block_closed = True
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                                    # Close all tool blocks
                                    for idx in range(1, anthropic_tool_index + 1):
                                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

                        except json.JSONDecodeError:
                            continue

        # Send final events with accumulated usage
        if not finished:
            # Close text block if not already closed
            if not text_block_closed:
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            for idx in range(1, anthropic_tool_index + 1):
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
            final_stop_reason = "end_turn"

        # Track usage for cost widget BEFORE final yield (estimate from text if tokens not available)
        output_text = "".join(accumulated_output)
        track_usage(original_request.model, input_tokens, output_tokens, input_text, output_text)

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': final_stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# Models for Anthropic API requests
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str

class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]

class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]

class SystemContent(BaseModel):
    type: Literal["text"]
    text: str

class Message(BaseModel):
    role: Literal["user", "assistant"] 
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]

class Tool(BaseModel):
    name: Optional[str] = None  # Optional for Anthropic-defined tools like web_search
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None  # Optional for Anthropic-defined tools
    type: Optional[str] = None  # For Anthropic-defined tools like web_search_20250305

class ThinkingConfig(BaseModel):
    enabled: bool = True

class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_field(cls, v, info): # Renamed to avoid conflict
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        if PREFERRED_PROVIDER == "anthropic":
            # Don't remap to big/small models, just add the prefix
            new_model = f"anthropic/{clean_v}"
            mapped = True

        # Map Haiku to SMALL_MODEL based on provider preference
        elif 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             # If no mapping occurred and no prefix exists, log warning or decide default
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name
    
    @field_validator('model')
    def validate_model_token_count(cls, v, info): # Renamed to avoid conflict
        # Use the same logic as MessagesRequest validator
        # NOTE: Pydantic validators might not share state easily if not class methods
        # Re-implementing the logic here for clarity, could be refactored
        original_model = v
        new_model = v # Default to original value

        logger.debug(f"📋 TOKEN COUNT VALIDATION: Original='{original_model}', Preferred='{PREFERRED_PROVIDER}', BIG='{BIG_MODEL}', SMALL='{SMALL_MODEL}'")

        # Remove provider prefixes for easier matching
        clean_v = v
        if clean_v.startswith('anthropic/'):
            clean_v = clean_v[10:]
        elif clean_v.startswith('openai/'):
            clean_v = clean_v[7:]
        elif clean_v.startswith('gemini/'):
            clean_v = clean_v[7:]

        # --- Mapping Logic --- START ---
        mapped = False
        # Map Haiku to SMALL_MODEL based on provider preference
        if 'haiku' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and SMALL_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{SMALL_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{SMALL_MODEL}"
                mapped = True

        # Map Sonnet to BIG_MODEL based on provider preference
        elif 'sonnet' in clean_v.lower():
            if PREFERRED_PROVIDER == "google" and BIG_MODEL in GEMINI_MODELS:
                new_model = f"gemini/{BIG_MODEL}"
                mapped = True
            else:
                new_model = f"openai/{BIG_MODEL}"
                mapped = True

        # Add prefixes to non-mapped models if they match known lists
        elif not mapped:
            if clean_v in GEMINI_MODELS and not v.startswith('gemini/'):
                new_model = f"gemini/{clean_v}"
                mapped = True # Technically mapped to add prefix
            elif clean_v in OPENAI_MODELS and not v.startswith('openai/'):
                new_model = f"openai/{clean_v}"
                mapped = True # Technically mapped to add prefix
        # --- Mapping Logic --- END ---

        if mapped:
            logger.debug(f"📌 TOKEN COUNT MAPPING: '{original_model}' ➡️ '{new_model}'")
        else:
             if not v.startswith(('openai/', 'gemini/', 'anthropic/')):
                 logger.warning(f"⚠️ No prefix or mapping rule for token count model: '{original_model}'. Using as is.")
             new_model = v # Ensure we return the original if no rule applied

        # Store the original model in the values dictionary
        values = info.data
        if isinstance(values, dict):
            values['original_model'] = original_model

        return new_model

class TokenCountResponse(BaseModel):
    input_tokens: int

class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Get request details
    method = request.method
    path = request.url.path
    
    # Log only basic request details at debug level
    logger.debug(f"Request: {method} {path}")
    
    # Process the request and get the response
    response = await call_next(request)
    
    return response

# Not using validation function as we're using the environment API key

def fix_empty_result(content) -> str:
    """Replace empty content with placeholder if FIX_EMPTY_TOOL_RESULTS is enabled."""
    if not FIX_EMPTY_TOOL_RESULTS:
        return content
    if content is None:
        return EMPTY_TOOL_RESULT_PLACEHOLDER
    if not isinstance(content, str):
        return content  # Don't modify non-strings
    if content == "" or content.strip() == "":
        return EMPTY_TOOL_RESULT_PLACEHOLDER
    return content

def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return fix_empty_result("No content provided")

    if isinstance(content, str):
        return fix_empty_result(content)
        
    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except:
                    result += "Unparseable content\n"
        return fix_empty_result(result.strip())

    if isinstance(content, dict):
        if content.get("type") == "text":
            return fix_empty_result(content.get("text", ""))
        try:
            return fix_empty_result(json.dumps(content))
        except:
            return fix_empty_result(str(content))

    # Fallback for any other type
    try:
        return fix_empty_result(str(content))
    except:
        return "Unparseable content"

def convert_anthropic_to_litellm(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic API request format to LiteLLM format (which follows OpenAI)."""
    # LiteLLM already handles Anthropic models when using the format model="anthropic/claude-3-opus-20240229"
    # So we just need to convert our Pydantic model to a dict in the expected format
    
    messages = []
    
    # Add system message if present
    if anthropic_request.system:
        # Handle different formats of system messages
        if isinstance(anthropic_request.system, str):
            # Simple string format
            messages.append({"role": "system", "content": anthropic_request.system})
        elif isinstance(anthropic_request.system, list):
            # List of content blocks
            system_text = ""
            for block in anthropic_request.system:
                if hasattr(block, 'type') and block.type == "text":
                    system_text += block.text + "\n\n"
                elif isinstance(block, dict) and block.get("type") == "text":
                    system_text += block.get("text", "") + "\n\n"
            
            if system_text:
                messages.append({"role": "system", "content": system_text.strip()})
    
    # Add conversation messages
    for idx, msg in enumerate(anthropic_request.messages):
        content = msg.content
        if isinstance(content, str):
            messages.append({"role": msg.role, "content": content})
        else:
            # Special handling for tool_result in user messages
            # OpenAI/LiteLLM format expects the assistant to call the tool, 
            # and the user's next message to include the result as plain text
            if msg.role == "user" and any(block.type == "tool_result" for block in content if hasattr(block, "type")):
                # For user messages with tool_result, split into separate messages
                text_content = ""
                
                # Extract all text parts and concatenate them
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content += block.text + "\n"
                        elif block.type == "tool_result":
                            # Add tool result as a message by itself - simulate the normal flow
                            tool_id = block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            
                            # Handle different formats of tool result content
                            result_content = ""
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    result_content = block.content
                                elif isinstance(block.content, list):
                                    # If content is a list of blocks, extract text from each
                                    for content_block in block.content:
                                        if hasattr(content_block, "type") and content_block.type == "text":
                                            result_content += content_block.text + "\n"
                                        elif isinstance(content_block, dict) and content_block.get("type") == "text":
                                            result_content += content_block.get("text", "") + "\n"
                                        elif isinstance(content_block, dict):
                                            # Handle any dict by trying to extract text or convert to JSON
                                            if "text" in content_block:
                                                result_content += content_block.get("text", "") + "\n"
                                            else:
                                                try:
                                                    result_content += json.dumps(content_block) + "\n"
                                                except:
                                                    result_content += str(content_block) + "\n"
                                elif isinstance(block.content, dict):
                                    # Handle dictionary content
                                    if block.content.get("type") == "text":
                                        result_content = block.content.get("text", "")
                                    else:
                                        try:
                                            result_content = json.dumps(block.content)
                                        except:
                                            result_content = str(block.content)
                                else:
                                    # Handle any other type by converting to string
                                    try:
                                        result_content = str(block.content)
                                    except:
                                        result_content = "Unparseable content"

                            # Apply empty result fix
                            result_content = fix_empty_result(result_content)

                            # In OpenAI format, tool results come from the user (rather than being content blocks)
                            text_content += f"Tool result for {tool_id}:\n{result_content}\n"
                
                # Add as a single user message with all the content
                messages.append({"role": "user", "content": text_content.strip()})
            else:
                # Regular handling for other message types
                processed_content = []
                for block in content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            processed_content.append({"type": "text", "text": block.text})
                        elif block.type == "image":
                            processed_content.append({"type": "image", "source": block.source})
                        elif block.type == "tool_use":
                            # Handle tool use blocks if needed
                            processed_content.append({
                                "type": "tool_use",
                                "id": block.id,
                                "name": block.name,
                                "input": block.input
                            })
                        elif block.type == "tool_result":
                            # Handle different formats of tool result content
                            processed_content_block = {
                                "type": "tool_result",
                                "tool_use_id": block.tool_use_id if hasattr(block, "tool_use_id") else ""
                            }

                            # Process the content field properly
                            if hasattr(block, "content"):
                                if isinstance(block.content, str):
                                    # If it's a simple string, create a text block for it
                                    text_val = fix_empty_result(block.content)
                                    processed_content_block["content"] = [{"type": "text", "text": text_val}]
                                elif isinstance(block.content, list):
                                    # If it's already a list of blocks, apply fix to text blocks
                                    fixed_content = []
                                    for item in block.content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            fixed_content.append({"type": "text", "text": fix_empty_result(item.get("text", ""))})
                                        else:
                                            fixed_content.append(item)
                                    processed_content_block["content"] = fixed_content if fixed_content else [{"type": "text", "text": fix_empty_result("")}]
                                else:
                                    # Default fallback
                                    processed_content_block["content"] = [{"type": "text", "text": fix_empty_result(str(block.content))}]
                            else:
                                # Default empty content
                                processed_content_block["content"] = [{"type": "text", "text": fix_empty_result("")}]

                            processed_content.append(processed_content_block)
                
                messages.append({"role": msg.role, "content": processed_content})
    
    # Cap max_tokens based on model limits
    # Use BIG_MODEL env var since request still has Anthropic model name at this point
    max_tokens = anthropic_request.max_tokens
    target_model = BIG_MODEL.lower() if BIG_MODEL else ""
    if "gpt-5" in target_model:
        # GPT-5.x supports up to 128k output tokens
        max_tokens = min(max_tokens, 128000)
    elif "gpt-4.1" in target_model:
        # GPT-4.1 supports up to 32k output tokens
        max_tokens = min(max_tokens, 32768)
    elif PREFERRED_PROVIDER in ["openai", "google"]:
        # Fallback for other OpenAI/Gemini models
        max_tokens = min(max_tokens, 16384)
        logger.debug(f"Capping max_tokens to 16384 for {PREFERRED_PROVIDER} model (original value: {anthropic_request.max_tokens})")
    
    # Create LiteLLM request dict
    litellm_request = {
        "model": anthropic_request.model,  # it understands "anthropic/claude-x" format
        "messages": messages,
        "max_completion_tokens": max_tokens,
        "temperature": anthropic_request.temperature,
        "stream": anthropic_request.stream,
    }

    # Request usage in streaming responses for OpenAI models
    if anthropic_request.stream and anthropic_request.model.startswith("openai/"):
        litellm_request["stream_options"] = {"include_usage": True}

    # Only include thinking field for Anthropic models
    if anthropic_request.thinking and anthropic_request.model.startswith("anthropic/"):
        litellm_request["thinking"] = anthropic_request.thinking

    # Add optional parameters if present
    if anthropic_request.stop_sequences:
        litellm_request["stop"] = anthropic_request.stop_sequences
    
    if anthropic_request.top_p:
        litellm_request["top_p"] = anthropic_request.top_p
    
    if anthropic_request.top_k:
        litellm_request["top_k"] = anthropic_request.top_k
    
    # Convert tools to OpenAI format
    if anthropic_request.tools:
        openai_tools = []
        is_gemini_model = anthropic_request.model.startswith("gemini/")

        for tool in anthropic_request.tools:
            # Convert to dict if it's a pydantic model
            if hasattr(tool, 'dict'):
                tool_dict = tool.dict()
            else:
                # Ensure tool_dict is a dictionary, handle potential errors if 'tool' isn't dict-like
                try:
                    tool_dict = dict(tool) if not isinstance(tool, dict) else tool
                except (TypeError, ValueError):
                     logger.error(f"Could not convert tool to dict: {tool}")
                     continue # Skip this tool if conversion fails

            # Clean the schema if targeting a Gemini model
            input_schema = tool_dict.get("input_schema", {})
            if is_gemini_model:
                 logger.debug(f"Cleaning schema for Gemini tool: {tool_dict.get('name')}")
                 input_schema = clean_gemini_schema(input_schema)

            # Create OpenAI-compatible function tool
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool_dict["name"],
                    "description": tool_dict.get("description", ""),
                    "parameters": input_schema # Use potentially cleaned schema
                }
            }
            openai_tools.append(openai_tool)

        litellm_request["tools"] = openai_tools
    
    # Convert tool_choice to OpenAI format if present
    if anthropic_request.tool_choice:
        if hasattr(anthropic_request.tool_choice, 'dict'):
            tool_choice_dict = anthropic_request.tool_choice.dict()
        else:
            tool_choice_dict = anthropic_request.tool_choice
            
        # Handle Anthropic's tool_choice format
        choice_type = tool_choice_dict.get("type")
        if choice_type == "auto":
            litellm_request["tool_choice"] = "auto"
        elif choice_type == "any":
            litellm_request["tool_choice"] = "any"
        elif choice_type == "tool" and "name" in tool_choice_dict:
            litellm_request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice_dict["name"]}
            }
        else:
            # Default to auto if we can't determine
            litellm_request["tool_choice"] = "auto"
    
    return litellm_request

def convert_litellm_to_anthropic(litellm_response: Union[Dict[str, Any], Any], 
                                 original_request: MessagesRequest) -> MessagesResponse:
    """Convert LiteLLM (OpenAI format) response to Anthropic API response format."""
    
    # Enhanced response extraction with better error handling
    try:
        
        # Handle ModelResponse object from LiteLLM
        if hasattr(litellm_response, 'choices') and hasattr(litellm_response, 'usage'):
            # Extract data from ModelResponse object directly
            choices = litellm_response.choices
            message = choices[0].message if choices and len(choices) > 0 else None
            content_text = message.content if message and hasattr(message, 'content') else ""
            tool_calls = message.tool_calls if message and hasattr(message, 'tool_calls') else None
            finish_reason = choices[0].finish_reason if choices and len(choices) > 0 else "stop"
            usage_info = litellm_response.usage
            response_id = getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}")
        else:
            # For backward compatibility - handle dict responses
            # If response is a dict, use it, otherwise try to convert to dict
            try:
                response_dict = litellm_response if isinstance(litellm_response, dict) else litellm_response.dict()
            except AttributeError:
                # If .dict() fails, try to use model_dump or __dict__ 
                try:
                    response_dict = litellm_response.model_dump() if hasattr(litellm_response, 'model_dump') else litellm_response.__dict__
                except AttributeError:
                    # Fallback - manually extract attributes
                    response_dict = {
                        "id": getattr(litellm_response, 'id', f"msg_{uuid.uuid4()}"),
                        "choices": getattr(litellm_response, 'choices', [{}]),
                        "usage": getattr(litellm_response, 'usage', {})
                    }
                    
            # Extract the content from the response dict
            choices = response_dict.get("choices", [{}])
            message = choices[0].get("message", {}) if choices and len(choices) > 0 else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", None)
            finish_reason = choices[0].get("finish_reason", "stop") if choices and len(choices) > 0 else "stop"
            usage_info = response_dict.get("usage", {})
            response_id = response_dict.get("id", f"msg_{uuid.uuid4()}")
        
        # Create content list for Anthropic format
        content = []
        
        # Add text content block if present (text might be None or empty for pure tool call responses)
        if content_text is not None and content_text != "":
            content.append({"type": "text", "text": content_text})
        
        # Add tool calls if present (tool_use in Anthropic format)
        # Always convert to proper tool_use blocks since client expects Anthropic API format
        if tool_calls:
            logger.debug(f"Processing tool calls: {tool_calls}")

            # Convert to list if it's not already
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]

            for idx, tool_call in enumerate(tool_calls):
                logger.debug(f"Processing tool call {idx}: {tool_call}")

                # Extract function data based on whether it's a dict or object
                if isinstance(tool_call, dict):
                    function = tool_call.get("function", {})
                    tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    name = function.get("name", "")
                    arguments = function.get("arguments", "{}")
                else:
                    function = getattr(tool_call, "function", None)
                    tool_id = getattr(tool_call, "id", f"toolu_{uuid.uuid4().hex[:24]}")
                    name = getattr(function, "name", "") if function else ""
                    arguments = getattr(function, "arguments", "{}") if function else "{}"

                # Convert string arguments to dict if needed
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments as JSON: {arguments}")
                        arguments = {"raw": arguments}

                logger.debug(f"Adding tool_use block: id={tool_id}, name={name}, input={arguments}")

                content.append({
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": arguments
                })
        
        # Get usage information - extract values safely from object or dict
        if isinstance(usage_info, dict):
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
        else:
            prompt_tokens = getattr(usage_info, "prompt_tokens", 0)
            completion_tokens = getattr(usage_info, "completion_tokens", 0)
        
        # Map OpenAI finish_reason to Anthropic stop_reason
        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "end_turn"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
            # Append warning to content so user sees it
            warning_text = "\n\n⚠️ **Hit max_tokens limit** - response was cut off. Say 'continue' to proceed."
            if content and content[-1].get("type") == "text":
                content[-1]["text"] += warning_text
            else:
                content.append({"type": "text", "text": warning_text})
        elif finish_reason == "tool_calls":
            stop_reason = "tool_use"
        else:
            stop_reason = "end_turn"  # Default

        # Make sure content is never empty
        if not content:
            content.append({"type": "text", "text": ""})
        
        # Create Anthropic-style response
        anthropic_response = MessagesResponse(
            id=response_id,
            model=original_request.model,
            role="assistant",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens
            )
        )
        
        return anthropic_response
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error converting response: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # In case of any error, create a fallback response
        return MessagesResponse(
            id=f"msg_{uuid.uuid4()}",
            model=original_request.model,
            role="assistant",
            content=[{"type": "text", "text": f"Error converting response: {str(e)}. Please check server logs."}],
            stop_reason="end_turn",
            usage=Usage(input_tokens=0, output_tokens=0)
        )

async def handle_streaming(response_generator, original_request: MessagesRequest):
    """Handle streaming responses from LiteLLM and convert to Anthropic format."""
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs
        
        message_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': original_request.model,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {
                    'input_tokens': 0,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                    'output_tokens': 0
                }
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_data)}\n\n"
        
        # Content block index for the first text block
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        
        # Send a ping to keep the connection alive (Anthropic does this)
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        
        tool_index = None
        current_tool_call = None
        tool_content = ""
        accumulated_text = ""  # Track accumulated text content
        text_sent = False  # Track if we've sent any text content
        text_block_closed = False  # Track if text block is closed
        input_tokens = 0
        output_tokens = 0
        has_sent_stop_reason = False
        last_tool_index = 0
        
        # Process each chunk
        async for chunk in response_generator:
            try:

                # Check if this is the end of the response with usage data
                if hasattr(chunk, 'usage') and chunk.usage is not None:
                    if hasattr(chunk.usage, 'prompt_tokens'):
                        input_tokens = chunk.usage.prompt_tokens
                    if hasattr(chunk.usage, 'completion_tokens'):
                        output_tokens = chunk.usage.completion_tokens
                
                # Handle text content
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    choice = chunk.choices[0]
                    
                    # Get the delta from the choice
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                    else:
                        # If no delta, try to get message
                        delta = getattr(choice, 'message', {})
                    
                    # Check for finish_reason to know when we're done
                    finish_reason = getattr(choice, 'finish_reason', None)
                    
                    # Process text content
                    delta_content = None
                    
                    # Handle different formats of delta content
                    if hasattr(delta, 'content'):
                        delta_content = delta.content
                    elif isinstance(delta, dict) and 'content' in delta:
                        delta_content = delta['content']
                    
                    # Accumulate text content
                    if delta_content is not None and delta_content != "":
                        accumulated_text += delta_content
                        
                        # Always emit text deltas if no tool calls started
                        if tool_index is None and not text_block_closed:
                            text_sent = True
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta_content}})}\n\n"
                    
                    # Process tool calls
                    delta_tool_calls = None
                    
                    # Handle different formats of tool calls
                    if hasattr(delta, 'tool_calls'):
                        delta_tool_calls = delta.tool_calls
                    elif isinstance(delta, dict) and 'tool_calls' in delta:
                        delta_tool_calls = delta['tool_calls']
                    
                    # Process tool calls if any
                    if delta_tool_calls:
                        # First tool call we've seen - need to handle text properly
                        if tool_index is None:
                            # If we've been streaming text, close that text block
                            if text_sent and not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # If we've accumulated text but not sent it, we need to emit it now
                            # This handles the case where the first delta has both text and a tool call
                            elif accumulated_text and not text_sent and not text_block_closed:
                                # Send the accumulated text
                                text_sent = True
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                                # Close the text block
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                            # Close text block even if we haven't sent anything - models sometimes emit empty text blocks
                            elif not text_block_closed:
                                text_block_closed = True
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                                
                        # Convert to list if it's not already
                        if not isinstance(delta_tool_calls, list):
                            delta_tool_calls = [delta_tool_calls]
                        
                        for tool_call in delta_tool_calls:
                            # Get the index of this tool call (for multiple tools)
                            current_index = None
                            if isinstance(tool_call, dict) and 'index' in tool_call:
                                current_index = tool_call['index']
                            elif hasattr(tool_call, 'index'):
                                current_index = tool_call.index
                            else:
                                current_index = 0
                            
                            # Check if this is a new tool or a continuation
                            if tool_index is None or current_index != tool_index:
                                # New tool call - create a new tool_use block
                                tool_index = current_index
                                last_tool_index += 1
                                anthropic_tool_index = last_tool_index
                                
                                # Extract function info
                                if isinstance(tool_call, dict):
                                    function = tool_call.get('function', {})
                                    name = function.get('name', '') if isinstance(function, dict) else ""
                                    tool_id = tool_call.get('id', f"toolu_{uuid.uuid4().hex[:24]}")
                                else:
                                    function = getattr(tool_call, 'function', None)
                                    name = getattr(function, 'name', '') if function else ''
                                    tool_id = getattr(tool_call, 'id', f"toolu_{uuid.uuid4().hex[:24]}")
                                
                                # Start a new tool_use block
                                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': anthropic_tool_index, 'content_block': {'type': 'tool_use', 'id': tool_id, 'name': name, 'input': {}}})}\n\n"
                                current_tool_call = tool_call
                                tool_content = ""
                            
                            # Extract function arguments
                            arguments = None
                            if isinstance(tool_call, dict) and 'function' in tool_call:
                                function = tool_call.get('function', {})
                                arguments = function.get('arguments', '') if isinstance(function, dict) else ''
                            elif hasattr(tool_call, 'function'):
                                function = getattr(tool_call, 'function', None)
                                arguments = getattr(function, 'arguments', '') if function else ''
                            
                            # If we have arguments, send them as a delta
                            if arguments:
                                # Try to detect if arguments are valid JSON or just a fragment
                                try:
                                    # If it's already a dict, use it
                                    if isinstance(arguments, dict):
                                        args_json = json.dumps(arguments)
                                    else:
                                        # Otherwise, try to parse it
                                        json.loads(arguments)
                                        args_json = arguments
                                except (json.JSONDecodeError, TypeError):
                                    # If it's a fragment, treat it as a string
                                    args_json = arguments
                                
                                # Add to accumulated tool content
                                tool_content += args_json if isinstance(args_json, str) else ""
                                
                                # Send the update
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': anthropic_tool_index, 'delta': {'type': 'input_json_delta', 'partial_json': args_json}})}\n\n"
                    
                    # Process finish_reason - end the streaming response
                    if finish_reason and not has_sent_stop_reason:
                        has_sent_stop_reason = True
                        
                        # Close any open tool call blocks
                        if tool_index is not None:
                            for i in range(1, last_tool_index + 1):
                                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
                        
                        # If we accumulated text but never sent or closed text block, do it now
                        if not text_block_closed:
                            if accumulated_text and not text_sent:
                                # Send the accumulated text
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': accumulated_text}})}\n\n"
                            # Close the text block
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Map OpenAI finish_reason to Anthropic stop_reason
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                            # Inject warning into response so user sees it
                            warning_text = "\n\n⚠️ **Hit max_tokens limit** - response was cut off. Say 'continue' to proceed."
                            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': warning_text}})}\n\n"
                        elif finish_reason == "tool_calls":
                            stop_reason = "tool_use"
                        elif finish_reason == "stop":
                            stop_reason = "end_turn"
                        
                        # Send message_delta with stop reason and usage
                        usage = {"output_tokens": output_tokens}

                        # Track usage for cost widget BEFORE final yields
                        track_usage(original_request.model, input_tokens, output_tokens)

                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': usage})}\n\n"

                        # Send message_stop event
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

                        # Send final [DONE] marker to match Anthropic's behavior
                        yield "data: [DONE]\n\n"
                        return
            except Exception as e:
                # Log error but continue processing other chunks
                logger.error(f"Error processing chunk: {str(e)}")
                continue
        
        # If we didn't get a finish reason, close any open blocks
        if not has_sent_stop_reason:
            # Close any open tool call blocks
            if tool_index is not None:
                for i in range(1, last_tool_index + 1):
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': i})}\n\n"
            
            # Close the text content block
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            
            # Send final message_delta with usage
            usage = {"output_tokens": output_tokens}

            # Track usage for cost widget BEFORE final yields
            track_usage(original_request.model, input_tokens, output_tokens)

            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': usage})}\n\n"

            # Send message_stop event
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            # Send final [DONE] marker to match Anthropic's behavior
            yield "data: [DONE]\n\n"

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_message = f"Error in streaming: {str(e)}\n\nFull traceback:\n{error_traceback}"
        logger.error(error_message)
        
        # Send error message_delta
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}, 'usage': {'output_tokens': 0}})}\n\n"
        
        # Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        
        # Send final [DONE] marker
        yield "data: [DONE]\n\n"

@app.post("/v1/messages")
async def create_message(
    request: MessagesRequest,
    raw_request: Request
):
    try:
        # print the body here
        body = await raw_request.body()
    
        # Parse the raw body as JSON since it's bytes
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        logger.debug(f"📊 PROCESSING REQUEST: Model={request.model}, Stream={request.stream}")

        # Check if this is a web search request
        # Parse tools from raw body to check for web_search tool type
        raw_tools = body_json.get("tools", [])
        if has_web_search_tool(raw_tools):
            # Determine the web search model
            web_search_model = get_openai_web_search_model(request.model)
            logger.info(f"🔍 WEB SEARCH REQUEST detected - routing to OpenAI Responses API with {web_search_model}")

            # Convert messages to input string for Responses API
            input_text = convert_anthropic_messages_to_input(request.messages, request.system)

            # Log the request
            num_tools = len(raw_tools) if raw_tools else 0
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                f"{web_search_model} (web_search)",
                len(request.messages),
                num_tools,
                200
            )

            if request.stream:
                # Handle streaming web search
                return StreamingResponse(
                    handle_web_search_streaming(
                        model=request.model,
                        input_text=input_text,
                        api_key=OPENAI_API_KEY,
                        original_request=request
                    ),
                    media_type="text/event-stream"
                )
            else:
                # Handle non-streaming web search
                try:
                    openai_response = await call_openai_responses_api_non_streaming(
                        model=request.model,
                        input_text=input_text,
                        api_key=OPENAI_API_KEY
                    )
                    anthropic_response = convert_openai_responses_to_anthropic(openai_response, request)
                    return JSONResponse(content=anthropic_response)
                except Exception as e:
                    logger.error(f"Error in web search: {e}")
                    raise HTTPException(status_code=500, detail=f"Web search error: {str(e)}")

        # Convert Anthropic request to LiteLLM format
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Determine which API key to use based on the model
        if request.model.startswith("openai/"):
            litellm_request["api_key"] = OPENAI_API_KEY
            # Use custom OpenAI base URL if configured
            if OPENAI_BASE_URL:
                litellm_request["api_base"] = OPENAI_BASE_URL
                # Add usecase header for llm-proxy.lan
                if "llm-proxy" in OPENAI_BASE_URL:
                    litellm_request["extra_headers"] = {"X-Usecase": "claude-hacky"}
                logger.debug(f"Using OpenAI API key and custom base URL {OPENAI_BASE_URL} for model: {request.model}")
            else:
                logger.debug(f"Using OpenAI API key for model: {request.model}")
        elif request.model.startswith("gemini/"):
            if USE_VERTEX_AUTH:
                litellm_request["vertex_project"] = VERTEX_PROJECT
                litellm_request["vertex_location"] = VERTEX_LOCATION
                litellm_request["custom_llm_provider"] = "vertex_ai"
                logger.debug(f"Using Gemini ADC with project={VERTEX_PROJECT}, location={VERTEX_LOCATION} and model: {request.model}")
            else:
                litellm_request["api_key"] = GEMINI_API_KEY
                logger.debug(f"Using Gemini API key for model: {request.model}")
        else:
            litellm_request["api_key"] = ANTHROPIC_API_KEY
            logger.debug(f"Using Anthropic API key for model: {request.model}")
        
        # For OpenAI models - modify request format to work with limitations
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"Processing OpenAI model request: {litellm_request['model']}")
            
            # For OpenAI models, we need to convert content blocks to simple strings
            # and handle other requirements
            for i, msg in enumerate(litellm_request["messages"]):
                # Special case - handle message content directly when it's a list of tool_result
                # This is a specific case we're seeing in the error
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.warning(f"Found message with only tool_result content - special handling required")
                        # Extract the content from all tool_result blocks
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            # Handle different formats of content
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        # Fall back to string representation of any dict
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        # Replace the list with extracted text
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        logger.warning(f"Converted tool_result to plain text: {all_text.strip()[:200]}...")
                        continue  # Skip normal processing for this message
                
                # 1. Handle content field - normal case
                if "content" in msg:
                    # Check if content is a list (content blocks)
                    if isinstance(msg["content"], list):
                        # Convert complex content blocks to simple string
                        text_content = ""
                        for block in msg["content"]:
                            if isinstance(block, dict):
                                # Handle different content block types
                                if block.get("type") == "text":
                                    text_content += block.get("text", "") + "\n"
                                
                                # Handle tool_result content blocks - extract nested text
                                elif block.get("type") == "tool_result":
                                    tool_id = block.get("tool_use_id", "unknown")
                                    text_content += f"[Tool Result ID: {tool_id}]\n"
                                    
                                    # Extract text from the tool_result content
                                    result_content = block.get("content", [])
                                    if isinstance(result_content, list):
                                        for item in result_content:
                                            if isinstance(item, dict) and item.get("type") == "text":
                                                text_content += item.get("text", "") + "\n"
                                            elif isinstance(item, dict):
                                                # Handle any dict by trying to extract text or convert to JSON
                                                if "text" in item:
                                                    text_content += item.get("text", "") + "\n"
                                                else:
                                                    try:
                                                        text_content += json.dumps(item) + "\n"
                                                    except:
                                                        text_content += str(item) + "\n"
                                    elif isinstance(result_content, dict):
                                        # Handle dictionary content
                                        if result_content.get("type") == "text":
                                            text_content += result_content.get("text", "") + "\n"
                                        else:
                                            try:
                                                text_content += json.dumps(result_content) + "\n"
                                            except:
                                                text_content += str(result_content) + "\n"
                                    elif isinstance(result_content, str):
                                        text_content += result_content + "\n"
                                    else:
                                        try:
                                            text_content += json.dumps(result_content) + "\n"
                                        except:
                                            text_content += str(result_content) + "\n"
                                
                                # Handle tool_use content blocks
                                elif block.get("type") == "tool_use":
                                    tool_name = block.get("name", "unknown")
                                    tool_id = block.get("id", "unknown")
                                    tool_input = json.dumps(block.get("input", {}))
                                    text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                                
                                # Handle image content blocks
                                elif block.get("type") == "image":
                                    text_content += "[Image content - not displayed in text format]\n"
                        
                        # Make sure content is never empty for OpenAI models
                        if not text_content.strip():
                            text_content = "..."
                        
                        litellm_request["messages"][i]["content"] = text_content.strip()
                    # Also check for None or empty string content
                    elif msg["content"] is None:
                        litellm_request["messages"][i]["content"] = "..." # Empty content not allowed
                
                # 2. Remove any fields OpenAI doesn't support in messages
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        logger.warning(f"Removing unsupported field from message: {key}")
                        del msg[key]
            
            # 3. Final validation - check for any remaining invalid values and dump full message details
            for i, msg in enumerate(litellm_request["messages"]):
                # Log the message format for debugging
                logger.debug(f"Message {i} format check - role: {msg.get('role')}, content type: {type(msg.get('content'))}")
                
                # If content is still a list or None, replace with placeholder
                if isinstance(msg.get("content"), list):
                    logger.warning(f"CRITICAL: Message {i} still has list content after processing: {json.dumps(msg.get('content'))}")
                    # Last resort - stringify the entire content as JSON
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    logger.warning(f"Message {i} has None content - replacing with placeholder")
                    litellm_request["messages"][i]["content"] = "..." # Fallback placeholder
        
        # Only log basic info about the request, not the full details
        logger.debug(f"Request for model: {litellm_request.get('model')}, stream: {litellm_request.get('stream', False)}")

        # Handle streaming mode
        if request.stream:
            num_tools = len(request.tools) if request.tools else 0

            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )

            # Bypass LiteLLM for custom base URLs (LiteLLM strips tools with custom api_base)
            # This includes llm-proxy.lan, studio.lan (Ollama), or any other custom endpoint
            if OPENAI_BASE_URL:
                return await stream_direct_to_llm_proxy(litellm_request, request, OPENAI_BASE_URL, OPENAI_API_KEY)

            # Use LiteLLM for other providers
            response_generator = await litellm.acompletion(**litellm_request)

            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            # Use LiteLLM for regular completion
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST", 
                raw_request.url.path, 
                display_model, 
                litellm_request.get('model'),
                len(litellm_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ RESPONSE RECEIVED: Model={litellm_request.get('model')}, Time={time.time() - start_time:.2f}s")
            
            # Convert LiteLLM response to Anthropic format
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)

            # Track usage for cost widget
            track_usage(
                litellm_request.get('model', 'unknown'),
                anthropic_response.usage.input_tokens,
                anthropic_response.usage.output_tokens
            )

            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Capture as much info as possible about the error
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        # Check for additional exception details in dictionaries
        if hasattr(e, '__dict__'):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ['args', '__traceback__']:
                    error_details[key] = str(value)
        
        # Helper function to safely serialize objects for JSON
        def sanitize_for_json(obj):
            """递归地清理对象使其可以JSON序列化"""
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return sanitize_for_json(obj.__dict__)
            elif hasattr(obj, 'text'):
                return str(obj.text)
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Log all error details with safe serialization
        sanitized_details = sanitize_for_json(error_details)
        logger.error(f"Error processing request: {json.dumps(sanitized_details, indent=2)}")
        
        # Format error for response
        error_message = f"Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        # Return detailed error
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: TokenCountRequest,
    raw_request: Request
):
    try:
        # Log the incoming token count request
        original_model = request.original_model or request.model
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        # Clean model name for capability check
        clean_model = request.model
        if clean_model.startswith("anthropic/"):
            clean_model = clean_model[len("anthropic/"):]
        elif clean_model.startswith("openai/"):
            clean_model = clean_model[len("openai/"):]
        
        # Convert the messages to a format LiteLLM can understand
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,  # Arbitrary value not used for token counting
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Use LiteLLM's token_counter function
        try:
            # Import token_counter function
            from litellm import token_counter
            
            # Log the request beautifully
            num_tools = len(request.tools) if request.tools else 0
            
            log_request_beautifully(
                "POST",
                raw_request.url.path,
                display_model,
                converted_request.get('model'),
                len(converted_request['messages']),
                num_tools,
                200  # Assuming success at this point
            )
            
            # Prepare token counter arguments (token counting is local, no API needed)
            token_counter_args = {
                "model": converted_request["model"],
                "messages": converted_request["messages"],
            }

            # Count tokens
            token_count = token_counter(**token_counter_args)
            
            # Return Anthropic-style response
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("Could not import token_counter from litellm")
            # Fallback to a simple approximation
            return TokenCountResponse(input_tokens=1000)  # Default fallback
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Anthropic Proxy for LiteLLM"}

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"
def log_request_beautifully(method, path, claude_model, openai_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful, twitter-friendly format showing Claude to OpenAI mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"
    
    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]
    
    # Extract just the OpenAI model name without provider prefix
    openai_display = openai_model
    if "/" in openai_display:
        openai_display = openai_display.split("/")[-1]
    openai_display = f"{Colors.GREEN}{openai_display}{Colors.RESET}"
    
    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
    
    # Format status code
    status_str = f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}" if status_code == 200 else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {openai_display} {tools_str} {messages_str}"
    
    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")
