"""
AURA MCP Server - AI-powered Kubernetes troubleshooting
Uses Ollama (free local LLM) instead of paid APIs
"""

import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
import json
from tools import KubernetesTools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AURA MCP Server", version="2.0.0")

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
K8S_NAMESPACE = os.getenv("K8S_NAMESPACE", "aura-system")

# Initialize Kubernetes tools
try:
    k8s_tools = KubernetesTools()
    logger.info("✅ Kubernetes client initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize Kubernetes client: {e}")
    k8s_tools = None


@app.on_event("startup")
async def check_ollama_model():
    """Ensure Ollama model is available"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                models_list = models_data.get("models", [])
                model_names = [m.get("name", "") for m in models_list]
                if not any(OLLAMA_MODEL in name for name in model_names):
                    logger.warning(f"Model {OLLAMA_MODEL} not found. Available: {model_names}")
                    logger.warning(f"Pull model with: docker exec aura-ollama ollama pull {OLLAMA_MODEL}")
                else:
                    logger.info(f"✅ Ollama model {OLLAMA_MODEL} ready")
    except Exception as e:
        logger.error(f"Failed to check Ollama models: {e}")


# Request/Response Models
class IssueAnalysisRequest(BaseModel):
    issue_id: str
    pod_name: str
    namespace: str
    issue_type: str
    severity: str
    description: str


class AnalysisResponse(BaseModel):
    action: str
    action_details: str
    reasoning: str
    confidence: float
    steps: Optional[list] = None


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AURA MCP Server"}


@app.post("/analyze")
async def analyze_issue(request: IssueAnalysisRequest) -> AnalysisResponse:
    """
    Analyze Kubernetes issue and recommend remediation action
    Uses Ollama (free local LLM) for AI analysis
    """
    try:
        logger.info(f"Analyzing issue: {request.namespace}/{request.pod_name} - {request.issue_type}")

        if not k8s_tools:
            logger.error("Kubernetes client not available")
            raise HTTPException(status_code=500, detail="Kubernetes client not available")

        # Gather pod context
        pod_info = k8s_tools.get_pod(request.namespace, request.pod_name)
        events = k8s_tools.get_events(request.namespace, request.pod_name, limit=3)
        logs = k8s_tools.get_pod_logs(request.namespace, request.pod_name, lines=10)
        deployment = k8s_tools.get_deployment_for_pod(request.namespace, request.pod_name)
        metrics = k8s_tools.get_pod_resource_usage(request.namespace, request.pod_name)

        # Build context for LLM
        context = f"""
ISSUE DETAILS:
- Pod: {request.namespace}/{request.pod_name}
- Issue Type: {request.issue_type}
- Severity: {request.severity}
- Description: {request.description}

POD STATUS:
- Status: {pod_info.get('status', 'Unknown')}
- Ready: {pod_info.get('ready', False)}
- Restart Count: {pod_info.get('restart_count', 0)}
- Node: {pod_info.get('node', 'Unknown')}

DEPLOYMENT INFO:
- Name: {deployment.get('name') if deployment else 'N/A'}
- Replicas: {deployment.get('replicas') if deployment else 'N/A'}
- Image: {deployment.get('image') if deployment else 'N/A'}

RESOURCE USAGE:
- CPU: {metrics.get('cpu_millicores', 'N/A')}m
- Memory: {metrics.get('memory_bytes', 'N/A')} bytes

RECENT EVENTS:
{json.dumps(events[:2], indent=2)}

RECENT LOGS:
{logs[:500]}
"""

        # Call Ollama for analysis
        prompt = f"""You are an expert Kubernetes SRE troubleshooting assistant.

{context}

Based on the pod information, logs, and events above, provide a JSON response with the following structure (NO other text):
{{
    "action": "restart_pod|increase_memory|increase_cpu|scale_deployment|clean_logs|drain_node",
    "action_details": "Brief description of what will be done",
    "reasoning": "Detailed explanation of why this action is recommended",
    "confidence": 0.85,
    "steps": [
        "Step 1: ...",
        "Step 2: ...",
    ]
}}

IMPORTANT: Respond ONLY with valid JSON, no other text."""

        response = await call_ollama(prompt)

        # Parse Ollama response
        analysis = parse_ollama_response(response)

        logger.info(f"✅ Analysis complete: recommended action '{analysis['action']}'")
        return AnalysisResponse(**analysis)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error analyzing issue: {e}")
        # Return a safe fallback
        return AnalysisResponse(
            action="restart_pod",
            action_details="Restart pod to attempt recovery",
            reasoning=f"Error during AI analysis: {str(e)}. Falling back to safe remediation.",
            confidence=0.5,
            steps=["Delete pod", "Kubernetes will recreate it"],
        )


@app.post("/get-pod-description")
async def get_pod_description(namespace: str, pod_name: str):
    """
    Get detailed description of a pod for manual investigation
    """
    try:
        if not k8s_tools:
            raise HTTPException(status_code=500, detail="Kubernetes client not available")

        description = k8s_tools.describe_pod(namespace, pod_name)
        return {"description": description}

    except Exception as e:
        logger.error(f"Failed to get pod description: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/namespace/{namespace}/overview")
async def get_namespace_overview(namespace: str):
    """
    Get overview of resources in a namespace
    """
    try:
        if not k8s_tools:
            raise HTTPException(status_code=500, detail="Kubernetes client not available")

        resources = k8s_tools.get_namespace_resources(namespace)
        return resources

    except Exception as e:
        logger.error(f"Failed to get namespace overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def call_ollama(prompt: str, retries: int = 2) -> str:
    """
    Call Ollama API for LLM inference with retry logic
    """
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.3, 
                    },
                )

                if response.status_code != 200:
                    logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                    if attempt < retries - 1:
                        await asyncio.sleep(2 ** attempt) 
                        continue
                    raise Exception(f"Ollama API error: {response.status_code}")

                result = response.json()
                return result.get("response", "")

        except httpx.ConnectError:
            logger.error(f"Failed to connect to Ollama at {OLLAMA_URL} (attempt {attempt + 1}/{retries})")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise Exception(f"Cannot connect to Ollama at {OLLAMA_URL}")
        except Exception as e:
            logger.error(f"Error calling Ollama (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
    
    return ""


def parse_ollama_response(response_text: str) -> dict:
    """
    Parse JSON response from Ollama
    Handles cases where Ollama may include extra text
    """
    try:
        # Try to find JSON in the response
        # Look for { } brackets
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            logger.warning(f"No JSON found in Ollama response: {response_text[:100]}")
            return get_fallback_response()

        json_str = response_text[start_idx:end_idx]
        analysis = json.loads(json_str)

        # Validate required fields
        required_fields = ["action", "action_details", "reasoning", "confidence"]
        for field in required_fields:
            if field not in analysis:
                logger.warning(f"Missing field '{field}' in Ollama response")
                return get_fallback_response()

        return analysis

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse Ollama JSON: {e}")
        return get_fallback_response()
    except Exception as e:
        logger.error(f"Error parsing Ollama response: {e}")
        return get_fallback_response()


def get_fallback_response() -> dict:
    """
    Return safe fallback response when AI analysis fails
    """
    return {
        "action": "restart_pod",
        "action_details": "Restart pod to attempt recovery",
        "reasoning": "Using safe fallback remediation. Please check logs for details.",
        "confidence": 0.5,
        "steps": ["Delete pod", "Kubernetes will recreate it"],
    }


@app.get("/models")
async def list_models():
    """
    List available models in Ollama
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": "Failed to fetch models"}
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MCP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
