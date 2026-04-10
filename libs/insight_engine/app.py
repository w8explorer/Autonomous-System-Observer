"""
Code Intelligence RAG - Streamlit Application
Main entry point for the application
"""

import os
import streamlit as st

from config.settings import APP_TITLE, APP_ICON, LAYOUT, INITIAL_SIDEBAR_STATE, DEFAULT_JAVA_PATH, LANGSMITH_ENABLED
from ui import CUSTOM_CSS, FOOTER_HTML, render_sidebar, render_message, render_process_path
from core import initialize_system
from evaluations.display import display_evaluation_results


# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=LAYOUT,
    initial_sidebar_state=INITIAL_SIDEBAR_STATE
)

# Apply custom CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vectorstore_loaded' not in st.session_state:
    st.session_state.vectorstore_loaded = False
if 'app' not in st.session_state:
    st.session_state.app = None
if 'java_path' not in st.session_state:
    st.session_state.java_path = DEFAULT_JAVA_PATH
if 'files_loaded' not in st.session_state:
    st.session_state.files_loaded = 0


def main():
    """Main application logic"""

    # Header
    st.markdown(f'<div class="main-header">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about your Java codebase in natural language</div>', unsafe_allow_html=True)

    # Sidebar
    java_path_input, show_intermediate, show_code, show_files, reload_requested, clear_requested = render_sidebar(
        st.session_state.java_path,
        st.session_state.files_loaded
    )

    # Handle path changes
    if java_path_input != st.session_state.java_path:
        st.session_state.java_path = java_path_input
        st.session_state.vectorstore_loaded = False

    # Handle reload request
    if reload_requested:
        st.session_state.vectorstore_loaded = False
        st.rerun()

    # Handle clear request
    if clear_requested:
        st.session_state.messages = []
        st.rerun()

    # Initialize system if needed
    if not st.session_state.vectorstore_loaded:
        if os.path.exists(st.session_state.java_path):
            try:
                app, retriever, files_count = initialize_system(st.session_state.java_path)
                if app and retriever:
                    st.session_state.app = app
                    st.session_state.vectorstore_loaded = True
                    st.session_state.files_loaded = files_count
                    st.success(f"✅ Code Intelligence System Ready! ({files_count} files indexed)")
            except Exception as e:
                st.error(f"❌ Error initializing system: {e}")
                st.info("💡 Make sure your .env file has OPENAI_API_KEY")
                st.stop()
        else:
            st.warning(f"⚠️ Directory not found: {st.session_state.java_path}")
            st.info("💡 Please create the directory and add your Java files, then click 'Load/Reload Codebase'")
            st.stop()

    # Display chat messages
    for message in st.session_state.messages:
        render_message(message, show_intermediate, show_files, show_code)

    # Chat input
    if prompt := st.chat_input("Ask about your Java code... (e.g., 'How does authentication work?')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing code..."):
                try:
                    # Run the code RAG
                    initial_state = {
                        "question": prompt,
                        "query_needs_improvement": False,
                        "retrieved_code": [],
                        "code_files": [],
                        "retrieved_documents": [],
                        "grading_scores": [],
                        "any_relevant": False,
                        "rewritten_query": "",
                        "generation": "",
                        "answer_quality_good": False,
                        "final_answer": "",
                        "intermediate_steps": [],
                        "retry_count": 0
                    }

                    result = st.session_state.app.invoke(initial_state)

                    # Display answer
                    answer = result.get("final_answer", "I couldn't generate an answer.")
                    st.markdown(answer)

                    # Determine and show path taken
                    path = render_process_path(result)
                    st.markdown(f'<div class="path-indicator">{path}</div>', unsafe_allow_html=True)

                    # Show intermediate steps
                    if show_intermediate:
                        with st.expander("🔍 View Process Steps", expanded=False):
                            for step in result.get("intermediate_steps", []):
                                st.markdown(f'<div class="step-box">{step}</div>', unsafe_allow_html=True)

                    # Show referenced files
                    if show_files and result.get("code_files"):
                        with st.expander("📁 Referenced Files", expanded=False):
                            unique_files = sorted(set(result.get("code_files", [])))
                            for filepath in unique_files:
                                st.markdown(f'<div class="file-reference">📄 {filepath}</div>', unsafe_allow_html=True)

                    # Show code snippets
                    if show_code and result.get("retrieved_code"):
                        with st.expander("💾 View Code Snippets", expanded=False):
                            codes = result.get("retrieved_code", [])
                            files = result.get("code_files", [])
                            scores = result.get("grading_scores", [])

                            for i, (code, filepath) in enumerate(zip(codes, files)):
                                score = scores[i]["score"] if i < len(scores) else "N/A"

                                st.markdown(f"**File:** `{filepath}`")
                                st.markdown(f"**Relevance:** {score}")
                                st.code(code[:500] + ("..." if len(code) > 500 else ""), language="java")
                                if i < len(codes) - 1:
                                    st.divider()

                    # Show evaluation results if available
                    if LANGSMITH_ENABLED and result.get("evaluation_results"):
                        display_evaluation_results(result.get("evaluation_results"))

                    # Save to session
                    message_data = {
                        "role": "assistant",
                        "content": answer,
                        "intermediate_steps": result.get("intermediate_steps", []),
                        "code_files": result.get("code_files", []),
                        "retrieved_code": result.get("retrieved_code", []),
                        "scores": [s["score"] for s in result.get("grading_scores", [])],
                        "evaluation_results": result.get("evaluation_results", {})
                    }
                    st.session_state.messages.append(message_data)

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Sorry, I encountered an error: {e}"
                    })

    # Footer
    st.divider()
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
