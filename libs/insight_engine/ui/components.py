"""
Streamlit UI components
"""

import streamlit as st


def render_sidebar(java_path: str, files_loaded: int):
    """
    Render the sidebar with configuration and controls

    Args:
        java_path: Current Java project path
        files_loaded: Number of files loaded

    Returns:
        Tuple of (new_java_path, show_intermediate, show_code, show_files, reload_requested, clear_requested)
    """
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Java project path input
        java_path_input = st.text_input(
            "Java Project Path",
            value=java_path,
            help="Path to your Java code directory"
        )

        reload_requested = st.button("🔄 Load/Reload Codebase", use_container_width=True)

        if files_loaded > 0:
            st.success(f"✅ {files_loaded} Java files loaded")

        st.divider()

        st.header("ℹ️ About")
        st.markdown("""
        This assistant helps you understand your Java codebase through natural language queries.

        ### Features:
        - 🔍 **Semantic Code Search**
        - 📁 **File References**
        - 💡 **Code Explanations**
        - ✅ **Quality Validation**
        - 🔄 **Query Rewriting** (up to 2 retries)

        ### Note:
        - **NO WEB SEARCH** - Only uses your codebase
        - Questions must relate to indexed code
        """)

        st.divider()

        st.header("🎛️ Display Options")
        show_intermediate = st.checkbox("Show process steps", value=True)
        show_code = st.checkbox("Show code snippets", value=True)
        show_files = st.checkbox("Show file references", value=True)

        st.divider()

        clear_requested = st.button("🗑️ Clear Chat History", use_container_width=True)

        # LangSmith Status
        st.divider()
        st.header("📊 Observability")
        try:
            from utils.langsmith_helper import display_langsmith_info
            langsmith_info = display_langsmith_info()
            st.markdown(langsmith_info)
        except Exception:
            # If LangSmith not configured, skip silently
            pass

    return java_path_input, show_intermediate, show_code, show_files, reload_requested, clear_requested


def render_message(message: dict, show_intermediate: bool, show_files: bool, show_code: bool):
    """
    Render a chat message with optional details

    Args:
        message: Message data dictionary
        show_intermediate: Whether to show intermediate steps
        show_files: Whether to show file references
        show_code: Whether to show code snippets
    """
    from config.settings import LANGSMITH_ENABLED
    from evaluations.display import display_evaluation_results

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if "intermediate_steps" in message and show_intermediate:
            with st.expander("🔍 View Process Steps"):
                for step in message["intermediate_steps"]:
                    st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

        if "code_files" in message and show_files and message["code_files"]:
            with st.expander("📁 Referenced Files"):
                unique_files = sorted(set(message["code_files"]))
                for filepath in unique_files:
                    st.markdown(f'<div class="file-reference">📄 {filepath}</div>', unsafe_allow_html=True)

        if "retrieved_code" in message and show_code and message["retrieved_code"]:
            with st.expander("💾 View Code Snippets"):
                for i, (code, filepath) in enumerate(zip(message["retrieved_code"], message["code_files"])):
                    score = message["scores"][i] if i < len(message.get("scores", [])) else "N/A"

                    st.markdown(f"**File:** `{filepath}`")
                    st.markdown(f"**Relevance:** {score}")
                    st.code(code[:500] + ("..." if len(code) > 500 else ""), language="java")
                    st.divider()

        # Show evaluation results if available
        if LANGSMITH_ENABLED and message.get("evaluation_results"):
            display_evaluation_results(message["evaluation_results"])


def render_process_path(result: dict) -> str:
    """
    Determine and format the processing path taken

    Args:
        result: Workflow result dictionary

    Returns:
        Formatted path string
    """
    if result.get("any_relevant"):
        return "🧠 Codebase → ✅ Relevant Code Found → 💬 Answer Generated"
    else:
        retry_count = result.get("retry_count", 0)
        if retry_count > 0:
            return f"🧠 Codebase → ❌ No Relevant Code → 🔄 Rewrite ({retry_count}x) → ⚠️ Still No Match"
        else:
            return "🧠 Codebase → ❌ No Relevant Code Found"
