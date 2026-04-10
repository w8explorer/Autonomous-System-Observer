"""
CSS styles for Streamlit UI
"""

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .code-box {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .file-reference {
        background-color: #e3f2fd;
        padding: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 0.5rem 0;
        font-family: monospace;
        cursor: pointer;
    }
    .file-reference:hover {
        background-color: #bbdefb;
    }
    .relevant-code {
        background-color: #d4edda;
        padding: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .irrelevant-code {
        background-color: #f8d7da;
        padding: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 0.5rem 0;
    }
    .path-indicator {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
    .step-box {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #6c757d;
    }
</style>
"""

FOOTER_HTML = """
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    💻 Code Intelligence RAG (No Web Search) | Powered by LangGraph • OpenAI GPT-4o • FAISS
</div>
"""
