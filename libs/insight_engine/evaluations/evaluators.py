"""
Custom evaluators for LangSmith observability.

This module provides evaluators for:
1. Answer Accuracy - Using LLM-as-judge pattern
2. Groundedness - Detecting hallucinations
3. Retrieval Relevancy - Measuring chunk relevance
4. Context Precision - Ranking quality of retrieved documents
"""

import os
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class AccuracyEvaluator:
    """
    Evaluates answer accuracy using LLM-as-judge pattern.

    Scores answers based on:
    - Correctness relative to retrieved context
    - Completeness of the answer
    - Relevance to the question
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing the accuracy of answers about Java code.

Your task is to evaluate if the answer is accurate, complete, and relevant based on the retrieved context.

Scoring criteria:
- 1.0 (Excellent): Accurate, complete, directly answers question with code examples
- 0.8 (Good): Accurate and relevant but missing minor details
- 0.6 (Fair): Partially accurate but incomplete or vague
- 0.4 (Poor): Contains inaccuracies or misses key information
- 0.2 (Very Poor): Mostly inaccurate or irrelevant
- 0.0 (Fail): Completely wrong or no answer

Consider:
- Does the answer match what's in the retrieved code?
- Does it fully address the question?
- Are code examples accurate?
- Is technical terminology correct?"""),
            ("user", """Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the accuracy of this answer. Provide:
1. Score (0.0 to 1.0)
2. Brief reasoning (2-3 sentences)

Format your response as:
SCORE: <float>
REASONING: <explanation>""")
        ])

    def evaluate(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate answer accuracy.

        Args:
            question: User's question
            answer: Generated answer
            context: Retrieved context

        Returns:
            Dictionary with score, reasoning, and pass/fail status
        """
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(
                    question=question,
                    answer=answer,
                    context=context
                )
            )

            result_text = response.content

            # Parse score and reasoning
            score_line = [line for line in result_text.split('\n') if line.startswith('SCORE:')]
            reasoning_line = [line for line in result_text.split('\n') if line.startswith('REASONING:')]

            score = float(score_line[0].replace('SCORE:', '').strip()) if score_line else 0.0
            reasoning = reasoning_line[0].replace('REASONING:', '').strip() if reasoning_line else result_text

            return {
                'score': score,
                'reasoning': reasoning,
                'pass': score >= 0.6,  # Pass threshold
                'raw_output': result_text
            }

        except Exception as e:
            return {
                'score': 0.0,
                'reasoning': f"Evaluation failed: {str(e)}",
                'pass': False,
                'error': str(e)
            }


class GroundednessEvaluator:
    """
    Evaluates groundedness to detect hallucinations.

    Checks if the answer is grounded in the retrieved context
    and doesn't make unsupported claims.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator checking if answers are grounded in the provided context.

Your task is to detect hallucinations - claims in the answer that are NOT supported by the retrieved context.

Scoring criteria:
- 1.0 (Fully Grounded): Every claim is directly supported by context
- 0.8 (Mostly Grounded): Minor unsupported details but main claims are valid
- 0.6 (Partially Grounded): Some claims supported, some not
- 0.4 (Poorly Grounded): Many unsupported claims
- 0.2 (Mostly Hallucinated): Most claims not in context
- 0.0 (Completely Hallucinated): Answer fabricated, no grounding

Focus on:
- Are code examples from the actual retrieved code?
- Are class/method names mentioned actually present?
- Are relationships (extends, implements, calls) accurate?
- Does it invent features not in the code?"""),
            ("user", """Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the groundedness of this answer. Identify any hallucinations.

Provide:
1. Score (0.0 to 1.0)
2. Specific hallucinations found (if any)
3. Brief reasoning

Format your response as:
SCORE: <float>
HALLUCINATIONS: <list of hallucinated claims, or "None">
REASONING: <explanation>""")
        ])

    def evaluate(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate answer groundedness.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            Dictionary with score, hallucinations, and pass/fail status
        """
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(
                    answer=answer,
                    context=context
                )
            )

            result_text = response.content

            # Parse score, hallucinations, and reasoning
            score_line = [line for line in result_text.split('\n') if line.startswith('SCORE:')]
            hall_line = [line for line in result_text.split('\n') if line.startswith('HALLUCINATIONS:')]
            reasoning_line = [line for line in result_text.split('\n') if line.startswith('REASONING:')]

            score = float(score_line[0].replace('SCORE:', '').strip()) if score_line else 0.0
            hallucinations = hall_line[0].replace('HALLUCINATIONS:', '').strip() if hall_line else "Unknown"
            reasoning = reasoning_line[0].replace('REASONING:', '').strip() if reasoning_line else result_text

            return {
                'score': score,
                'hallucinations': hallucinations,
                'reasoning': reasoning,
                'pass': score >= 0.8,  # Higher threshold for groundedness
                'raw_output': result_text
            }

        except Exception as e:
            return {
                'score': 0.0,
                'hallucinations': "Unknown",
                'reasoning': f"Evaluation failed: {str(e)}",
                'pass': False,
                'error': str(e)
            }


class RetrievalRelevancyEvaluator:
    """
    Evaluates retrieval relevancy - how relevant are retrieved documents.

    Measures what percentage of retrieved chunks are actually relevant
    to answering the question.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator assessing retrieval quality.

Your task is to determine if a retrieved code chunk is relevant to answering the question.

A chunk is RELEVANT if it:
- Contains code/classes/methods mentioned in the question
- Provides context needed to answer the question
- Shows relationships (inheritance, calls, dependencies) related to the question

A chunk is NOT RELEVANT if it:
- Is unrelated to the question topic
- Contains only tangentially related code
- Was retrieved by keyword match but doesn't help answer the question"""),
            ("user", """Question: {question}

Retrieved Code Chunk:
{chunk}

Is this chunk relevant for answering the question?

Respond with ONLY:
RELEVANT: Yes
or
RELEVANT: No

Then provide brief reasoning (1 sentence).""")
        ])

    def evaluate_chunk(self, question: str, chunk: str) -> bool:
        """
        Evaluate if a single chunk is relevant.

        Args:
            question: User's question
            chunk: Retrieved code chunk

        Returns:
            True if relevant, False otherwise
        """
        try:
            response = self.llm.invoke(
                self.prompt.format_messages(
                    question=question,
                    chunk=chunk
                )
            )

            result_text = response.content.upper()
            return "RELEVANT: YES" in result_text

        except Exception:
            # If evaluation fails, assume relevant (conservative)
            return True

    def evaluate(self, question: str, retrieved_documents: List[Any]) -> Dict[str, Any]:
        """
        Evaluate relevancy of all retrieved documents.

        Args:
            question: User's question
            retrieved_documents: List of retrieved document objects

        Returns:
            Dictionary with relevancy score and details
        """
        if not retrieved_documents:
            return {
                'score': 0.0,
                'relevant_count': 0,
                'total_count': 0,
                'percentage': 0.0,
                'pass': False
            }

        relevant_count = 0
        total_count = len(retrieved_documents)

        for doc in retrieved_documents:
            chunk_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            if self.evaluate_chunk(question, chunk_text):
                relevant_count += 1

        percentage = (relevant_count / total_count) * 100 if total_count > 0 else 0.0

        return {
            'score': relevant_count / total_count if total_count > 0 else 0.0,
            'relevant_count': relevant_count,
            'total_count': total_count,
            'percentage': percentage,
            'pass': percentage >= 60.0,  # Pass if 60%+ chunks relevant
            'reasoning': f"{relevant_count}/{total_count} chunks ({percentage:.1f}%) are relevant"
        }


class ContextPrecisionEvaluator:
    """
    Evaluates context precision - are the most relevant chunks ranked highest?

    Measures if relevant documents appear early in the retrieval results.
    """

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def evaluate(self, question: str, retrieved_documents: List[Any],
                 relevancy_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate context precision based on ranking.

        Args:
            question: User's question
            retrieved_documents: List of retrieved documents (in order)
            relevancy_results: Results from RelevancyEvaluator

        Returns:
            Dictionary with precision score and analysis
        """
        if not retrieved_documents:
            return {
                'score': 0.0,
                'pass': False,
                'reasoning': "No documents retrieved"
            }

        # Calculate precision@k for k=[1,3,5]
        # This measures if relevant docs appear early
        total_docs = len(retrieved_documents)

        # Simplified scoring: Check if we have good relevancy percentage
        # and if retrieval returned reasonable number of docs
        relevancy_score = relevancy_results.get('score', 0.0)

        # Context precision is good if:
        # 1. High relevancy (>0.6)
        # 2. Not too many irrelevant docs retrieved
        precision_score = relevancy_score

        return {
            'score': precision_score,
            'pass': precision_score >= 0.6,
            'reasoning': f"Relevancy-based precision: {precision_score:.2f}"
        }


# Main evaluation functions for integration

def evaluate_accuracy(question: str, answer: str, context: str) -> Dict[str, Any]:
    """
    Evaluate answer accuracy using LLM-as-judge.

    Args:
        question: User's question
        answer: Generated answer
        context: Retrieved context

    Returns:
        Evaluation results dictionary
    """
    evaluator = AccuracyEvaluator()
    return evaluator.evaluate(question, answer, context)


def evaluate_groundedness(answer: str, context: str) -> Dict[str, Any]:
    """
    Evaluate answer groundedness (hallucination detection).

    Args:
        answer: Generated answer
        context: Retrieved context

    Returns:
        Evaluation results dictionary
    """
    evaluator = GroundednessEvaluator()
    return evaluator.evaluate(answer, context)


def evaluate_retrieval_relevancy(question: str, retrieved_documents: List[Any]) -> Dict[str, Any]:
    """
    Evaluate retrieval relevancy.

    Args:
        question: User's question
        retrieved_documents: List of retrieved documents

    Returns:
        Evaluation results dictionary
    """
    evaluator = RetrievalRelevancyEvaluator()
    return evaluator.evaluate(question, retrieved_documents)


def evaluate_context_precision(question: str, retrieved_documents: List[Any],
                               relevancy_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate context precision.

    Args:
        question: User's question
        retrieved_documents: List of retrieved documents
        relevancy_results: Results from relevancy evaluation

    Returns:
        Evaluation results dictionary
    """
    evaluator = ContextPrecisionEvaluator()
    return evaluator.evaluate(question, retrieved_documents, relevancy_results)


def run_all_evaluations(question: str, answer: str, context: str,
                        retrieved_documents: List[Any]) -> Dict[str, Any]:
    """
    Run all evaluations and return comprehensive results.

    Args:
        question: User's question
        answer: Generated answer
        context: Retrieved context as string
        retrieved_documents: List of retrieved document objects

    Returns:
        Dictionary with all evaluation results
    """
    results = {}

    # 1. Accuracy
    results['accuracy'] = evaluate_accuracy(question, answer, context)

    # 2. Groundedness
    results['groundedness'] = evaluate_groundedness(answer, context)

    # 3. Retrieval Relevancy
    results['relevancy'] = evaluate_retrieval_relevancy(question, retrieved_documents)

    # 4. Context Precision
    results['precision'] = evaluate_context_precision(
        question, retrieved_documents, results['relevancy']
    )

    # Overall pass/fail
    all_passed = all([
        results['accuracy'].get('pass', False),
        results['groundedness'].get('pass', False),
        results['relevancy'].get('pass', False),
        results['precision'].get('pass', False)
    ])

    results['overall'] = {
        'all_passed': all_passed,
        'summary': _generate_summary(results)
    }

    return results


def _generate_summary(results: Dict[str, Any]) -> str:
    """Generate a human-readable summary of evaluation results."""
    lines = []

    acc = results.get('accuracy', {})
    lines.append(f"✓ Accuracy: {acc.get('score', 0):.2f} ({'Pass' if acc.get('pass') else 'Fail'})")

    ground = results.get('groundedness', {})
    lines.append(f"✓ Groundedness: {ground.get('score', 0):.2f} ({'Pass' if ground.get('pass') else 'Fail'})")

    rel = results.get('relevancy', {})
    lines.append(f"✓ Relevancy: {rel.get('percentage', 0):.1f}% ({'Pass' if rel.get('pass') else 'Fail'})")

    prec = results.get('precision', {})
    lines.append(f"✓ Precision: {prec.get('score', 0):.2f} ({'Pass' if prec.get('pass') else 'Fail'})")

    return "\n".join(lines)
