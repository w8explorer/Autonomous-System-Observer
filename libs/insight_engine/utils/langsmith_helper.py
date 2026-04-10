"""
LangSmith integration helper utilities
"""

import os
from typing import Dict, Any


def get_langsmith_status() -> Dict[str, Any]:
    """
    Get LangSmith configuration status

    Returns:
        Dictionary with LangSmith status and configuration
    """
    from config.settings import (
        LANGSMITH_ENABLED,
        LANGSMITH_API_KEY,
        LANGSMITH_PROJECT,
        ENVIRONMENT
    )

    is_configured = bool(LANGSMITH_API_KEY and LANGSMITH_ENABLED)

    return {
        "enabled": LANGSMITH_ENABLED,
        "configured": is_configured,
        "project": LANGSMITH_PROJECT,
        "environment": ENVIRONMENT,
        "has_api_key": bool(LANGSMITH_API_KEY),
        "dashboard_url": f"https://smith.langchain.com/o/default/projects/p/{LANGSMITH_PROJECT}" if is_configured else None
    }


def display_langsmith_info() -> str:
    """
    Generate display text for LangSmith status

    Returns:
        Formatted status message
    """
    status = get_langsmith_status()

    if status["configured"]:
        return f"""
✅ **LangSmith Observability: ENABLED**
- Project: `{status['project']}`
- Environment: `{status['environment']}`
- Dashboard: [View Traces]({status['dashboard_url']})

All queries are being traced for evaluation and monitoring.
"""
    elif status["enabled"] and not status["has_api_key"]:
        return """
⚠️ **LangSmith: Enabled but not configured**
- Missing API key in .env file
- Add LANGCHAIN_API_KEY to enable tracing
"""
    else:
        return """
ℹ️ **LangSmith: Disabled**
- Set LANGCHAIN_TRACING_V2=true in .env to enable
- Sign up at https://smith.langchain.com
"""


def get_trace_url(run_id: str = None) -> str:
    """
    Generate LangSmith trace URL for a specific run

    Args:
        run_id: LangSmith run ID

    Returns:
        URL to view trace in LangSmith dashboard
    """
    from config.settings import LANGSMITH_PROJECT

    if run_id:
        return f"https://smith.langchain.com/o/default/projects/p/{LANGSMITH_PROJECT}/r/{run_id}"
    return f"https://smith.langchain.com/o/default/projects/p/{LANGSMITH_PROJECT}"


def get_feedback_stats(limit: int = 100) -> Dict[str, Any]:
    """
    Get statistics on evaluation feedback from LangSmith.

    Args:
        limit: Number of recent runs to analyze

    Returns:
        Dictionary with feedback statistics
    """
    from config.settings import LANGSMITH_ENABLED, LANGSMITH_PROJECT

    if not LANGSMITH_ENABLED:
        return {"error": "LangSmith not enabled"}

    try:
        from langsmith import Client

        client = Client()

        # Get recent runs
        runs = list(client.list_runs(
            project_name=LANGSMITH_PROJECT,
            limit=limit
        ))

        if not runs:
            return {"error": "No runs found", "runs_count": 0}

        # Collect feedback stats
        stats = {
            "total_runs": len(runs),
            "runs_with_feedback": 0,
            "accuracy_scores": [],
            "groundedness_scores": [],
            "relevancy_scores": [],
            "precision_scores": []
        }

        for run in runs:
            # Get feedback for this run
            feedback = list(client.list_feedback(run_ids=[run.id]))

            if feedback:
                stats["runs_with_feedback"] += 1

                for fb in feedback:
                    if fb.key == "accuracy" and fb.score is not None:
                        stats["accuracy_scores"].append(fb.score)
                    elif fb.key == "groundedness" and fb.score is not None:
                        stats["groundedness_scores"].append(fb.score)
                    elif fb.key == "retrieval_relevancy" and fb.score is not None:
                        stats["relevancy_scores"].append(fb.score)
                    elif fb.key == "context_precision" and fb.score is not None:
                        stats["precision_scores"].append(fb.score)

        # Calculate averages
        if stats["accuracy_scores"]:
            stats["avg_accuracy"] = sum(stats["accuracy_scores"]) / len(stats["accuracy_scores"])
        if stats["groundedness_scores"]:
            stats["avg_groundedness"] = sum(stats["groundedness_scores"]) / len(stats["groundedness_scores"])
        if stats["relevancy_scores"]:
            stats["avg_relevancy"] = sum(stats["relevancy_scores"]) / len(stats["relevancy_scores"])
        if stats["precision_scores"]:
            stats["avg_precision"] = sum(stats["precision_scores"]) / len(stats["precision_scores"])

        return stats

    except Exception as e:
        return {"error": str(e)}
