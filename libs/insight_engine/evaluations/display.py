"""
Display utilities for evaluation results in Streamlit UI.
"""

import streamlit as st
from typing import Dict, Any


def display_evaluation_results(eval_results: Dict[str, Any]):
    """
    Display evaluation results in Streamlit UI.

    Args:
        eval_results: Dictionary containing evaluation results
    """
    if not eval_results:
        return

    st.divider()
    st.subheader("📊 Evaluation Metrics")

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    # 1. Accuracy
    accuracy = eval_results.get('accuracy', {})
    with col1:
        score = accuracy.get('score', 0.0)
        passed = accuracy.get('pass', False)
        st.metric(
            "Accuracy",
            f"{score:.2f}",
            delta="Pass" if passed else "Fail",
            delta_color="normal" if passed else "inverse"
        )

    # 2. Groundedness
    groundedness = eval_results.get('groundedness', {})
    with col2:
        score = groundedness.get('score', 0.0)
        passed = groundedness.get('pass', False)
        st.metric(
            "Groundedness",
            f"{score:.2f}",
            delta="Pass" if passed else "Fail",
            delta_color="normal" if passed else "inverse"
        )

    # 3. Relevancy
    relevancy = eval_results.get('relevancy', {})
    with col3:
        percentage = relevancy.get('percentage', 0.0)
        passed = relevancy.get('pass', False)
        st.metric(
            "Relevancy",
            f"{percentage:.1f}%",
            delta="Pass" if passed else "Fail",
            delta_color="normal" if passed else "inverse"
        )

    # 4. Precision
    precision = eval_results.get('precision', {})
    with col4:
        score = precision.get('score', 0.0)
        passed = precision.get('pass', False)
        st.metric(
            "Precision",
            f"{score:.2f}",
            delta="Pass" if passed else "Fail",
            delta_color="normal" if passed else "inverse"
        )

    # Expandable details
    with st.expander("📝 Evaluation Details"):
        # Accuracy details
        st.markdown("**🎯 Accuracy Evaluation**")
        st.markdown(f"- Score: {accuracy.get('score', 0):.2f}")
        st.markdown(f"- Status: {'✅ Pass' if accuracy.get('pass') else '❌ Fail'}")
        st.markdown(f"- Reasoning: {accuracy.get('reasoning', 'N/A')}")
        st.markdown("")

        # Groundedness details
        st.markdown("**🔍 Groundedness Evaluation**")
        st.markdown(f"- Score: {groundedness.get('score', 0):.2f}")
        st.markdown(f"- Status: {'✅ Pass' if groundedness.get('pass') else '❌ Fail'}")
        st.markdown(f"- Hallucinations: {groundedness.get('hallucinations', 'N/A')}")
        st.markdown(f"- Reasoning: {groundedness.get('reasoning', 'N/A')}")
        st.markdown("")

        # Relevancy details
        st.markdown("**📚 Retrieval Relevancy Evaluation**")
        st.markdown(f"- Relevant: {relevancy.get('relevant_count', 0)}/{relevancy.get('total_count', 0)} chunks")
        st.markdown(f"- Percentage: {relevancy.get('percentage', 0):.1f}%")
        st.markdown(f"- Status: {'✅ Pass' if relevancy.get('pass') else '❌ Fail'}")
        st.markdown("")

        # Precision details
        st.markdown("**🎲 Context Precision Evaluation**")
        st.markdown(f"- Score: {precision.get('score', 0):.2f}")
        st.markdown(f"- Status: {'✅ Pass' if precision.get('pass') else '❌ Fail'}")
        st.markdown(f"- Reasoning: {precision.get('reasoning', 'N/A')}")

    # Overall status
    overall = eval_results.get('overall', {})
    if overall.get('all_passed'):
        st.success("✅ All evaluation metrics passed!")
    else:
        st.warning("⚠️ Some evaluation metrics did not pass. Review details above.")


def get_evaluation_summary(eval_results: Dict[str, Any]) -> str:
    """
    Get a brief text summary of evaluation results.

    Args:
        eval_results: Dictionary containing evaluation results

    Returns:
        String summary
    """
    if not eval_results:
        return "No evaluation results available"

    lines = []

    # Accuracy
    accuracy = eval_results.get('accuracy', {})
    acc_status = "✅" if accuracy.get('pass') else "❌"
    lines.append(f"{acc_status} Accuracy: {accuracy.get('score', 0):.2f}")

    # Groundedness
    groundedness = eval_results.get('groundedness', {})
    ground_status = "✅" if groundedness.get('pass') else "❌"
    lines.append(f"{ground_status} Groundedness: {groundedness.get('score', 0):.2f}")

    # Relevancy
    relevancy = eval_results.get('relevancy', {})
    rel_status = "✅" if relevancy.get('pass') else "❌"
    lines.append(f"{rel_status} Relevancy: {relevancy.get('percentage', 0):.1f}%")

    # Precision
    precision = eval_results.get('precision', {})
    prec_status = "✅" if precision.get('pass') else "❌"
    lines.append(f"{prec_status} Precision: {precision.get('score', 0):.2f}")

    return "\n".join(lines)
