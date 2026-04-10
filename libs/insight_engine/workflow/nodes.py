"""
LangGraph workflow nodes
"""

from langchain_core.messages import HumanMessage

from models.state import GraphState
from utils.relationship_filter import filter_by_relationships


class WorkflowNodes:
    """Collection of workflow nodes for the RAG system"""

    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever

    def query_quality_node(self, state: GraphState) -> GraphState:
        """Check if query needs improvement before retrieval"""
        question = state["question"]

        # Simple heuristic checks first
        word_count = len(question.split())
        has_typo_indicators = any(
            indicator in question.lower()
            for indicator in ['passowrd', 'athentication', 'valdiation']
        )

        # If very short or has obvious issues, mark for improvement
        if word_count <= 1 or has_typo_indicators:
            needs_improvement = True
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                f"⚠️ Query quality check: Too short or unclear ('{question}')"
            ]
        else:
            # Ask LLM for more nuanced assessment
            prompt = f"""Analyze if this question is clear enough to search an EXISTING Java codebase.

Question: "{question}"

Context: This is for searching an existing codebase, not for tutorials or how-to guides.

Consider NEEDS IMPROVEMENT if:
- Single vague word without context (e.g., just "user" or "service")
- Obvious typos or misspellings
- Completely unclear what to search for
- Missing basic context about what aspect to find

Consider GOOD (even if brief) if:
- Class name mentioned (e.g., "AuthenticationService") - single class names are FINE
- Method name with some context (e.g., "login method")
- Feature name that can be searched (e.g., "payment processing")
- Relationship query (e.g., "what calls User")
- Any specific code element is mentioned

NOTE: Single class/method names are ACCEPTABLE - they will be expanded during rewriting.

Does this question need improvement before searching?
Answer only: yes or no"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            needs_improvement = "yes" in response.content.lower()

            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                f"Query quality check: {'⚠️ Needs improvement' if needs_improvement else '✅ Good quality'}"
            ]

        state["query_needs_improvement"] = needs_improvement
        return state

    def retrieve_node(self, state: GraphState) -> GraphState:
        """Retrieve relevant code chunks from vector store"""
        rewritten = state.get("rewritten_query", "")
        query = rewritten if rewritten and rewritten.strip() else state["question"]
        question = state["question"]

        # Get documents from vector store
        documents = self.retriever.invoke(query)

        # Apply relationship filtering (Hybrid Lite!)
        filtered_docs = filter_by_relationships(documents, question)

        # Check if filtering helped
        if len(filtered_docs) < len(documents):
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                f"📚 Retrieved {len(documents)} chunks → 🔍 Filtered to {len(filtered_docs)} by relationships"
            ]
        else:
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                f"📚 Retrieved {len(documents)} code chunks from {len(set([d.metadata.get('source', 'Unknown') for d in documents]))} files"
            ]

        state["retrieved_code"] = [doc.page_content for doc in filtered_docs]
        state["code_files"] = [doc.metadata.get("source", "Unknown") for doc in filtered_docs]
        state["retrieved_documents"] = filtered_docs  # Store full Document objects for evaluators

        return state

    def grade_documents_node(self, state: GraphState) -> GraphState:
        """Grade relevance of retrieved documents"""
        question = state["question"]
        code_chunks = state["retrieved_code"]

        grading_scores = []
        for i, code in enumerate(code_chunks):
            prompt = f"""Grade if this code is relevant to the question.

Question: {question}

Code:
{code[:800]}

Is this code relevant to answering the question?
Consider relevant if the code:
- Implements the functionality being asked about
- Contains related classes, methods, or variables
- Shows the architecture or design pattern in question
- Has relevant imports, annotations, or configurations

Answer only: yes or no"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            score = "yes" if "yes" in response.content.lower() else "no"
            grading_scores.append({"doc_index": i, "score": score})

        state["grading_scores"] = grading_scores
        state["any_relevant"] = any(g["score"] == "yes" for g in grading_scores)

        relevant_count = sum(1 for g in grading_scores if g["score"] == "yes")
        state["intermediate_steps"] = state.get("intermediate_steps", []) + [
            f"✅ {relevant_count}/{len(grading_scores)} code chunks marked relevant"
        ]
        return state

    def rewrite_query_node(self, state: GraphState) -> GraphState:
        """Rewrite query for better search results"""
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        prompt = f"""Rewrite this question to search an EXISTING Java codebase more effectively.

Original question: {question}

IMPORTANT: This is about finding and understanding EXISTING code, not building new code.

Rewrite the query to be:
- Discovery-focused: Use words like "find", "show", "locate", "implementation of" instead of "how to implement"
- Codebase-specific: Focus on finding what EXISTS in the code, not generic tutorials
- Analysis-oriented: Use terms like "class definition", "method implementation", "usage", "callers", "dependencies"
- Concrete: Add technical terms (class, method, interface, annotation) but DON'T assume features that may not exist
- Relationship-aware: Include words like "where used", "dependencies", "related code" when relevant

Examples:
- "AuthenticationService" → "Find AuthenticationService class implementation, its methods, and where it is used"
- "login" → "Find login method implementation and authentication flow in the codebase"
- "User" → "Find User class definition, its fields, methods, and relationships with other classes"

Provide ONLY the rewritten question, nothing else."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        rewritten = response.content.strip()

        state["rewritten_query"] = rewritten
        state["retry_count"] = retry_count + 1
        state["intermediate_steps"] = state.get("intermediate_steps", []) + [
            f"✏️ Query rewritten (attempt {state['retry_count']}): \"{rewritten}\""
        ]
        return state

    def generate_node(self, state: GraphState) -> GraphState:
        """Generate answer from retrieved code"""
        question = state["question"]

        if state.get("retrieved_code") and state.get("any_relevant"):
            code_chunks = state["retrieved_code"]
            file_paths = state["code_files"]

            context_parts = []
            for code, filepath in zip(code_chunks, file_paths):
                context_parts.append(f"File: {filepath}\n{code}")
            context = "\n\n---\n\n".join(context_parts)

            prompt = f"""Answer this code-related question using the provided code snippets.

Question: {question}

Code Context:
{context}

Provide a clear answer that:
1. Directly answers the question
2. References specific file paths and code elements (classes, methods, variables)
3. Explains the code logic if relevant
4. Mentions related code or patterns if helpful

Answer:"""

            response = self.llm.invoke([HumanMessage(content=prompt)])
            state["generation"] = response.content
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                "💬 Generated answer from codebase"
            ]
        else:
            state["generation"] = "I couldn't find relevant information in the codebase to answer this question. Please make sure your question relates to the indexed Java code."
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                "❌ No relevant code found - cannot answer"
            ]

        return state

    def self_reflection_node(self, state: GraphState) -> GraphState:
        """Evaluate quality of generated answer"""
        generation = state["generation"]
        question = state["question"]

        # Skip reflection if no answer was generated
        if "couldn't find relevant information" in generation:
            state["answer_quality_good"] = False
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                "🤔 Self-reflection: Skipped (no answer generated)"
            ]
            return state

        prompt = f"""Evaluate if this answer about code is accurate and well-supported.

Question: {question}
Answer: {generation}

Check if the answer:
- Actually addresses the question
- References specific code elements (files, classes, methods)
- Makes logical sense for a Java codebase
- Doesn't make unsupported claims

Is this a good quality answer?
Respond: yes or no"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        is_good = "yes" in response.content.lower()

        state["answer_quality_good"] = is_good
        state["intermediate_steps"] = state.get("intermediate_steps", []) + [
            f"🤔 Self-reflection: {'✅ Quality check passed' if is_good else '⚠️ Quality concerns detected'}"
        ]
        return state

    def evaluation_node(self, state: GraphState) -> GraphState:
        """
        Evaluate the generated answer using custom evaluators.
        Only runs if LangSmith is enabled.
        """
        from config.settings import LANGSMITH_ENABLED

        # Skip evaluation if LangSmith not enabled
        if not LANGSMITH_ENABLED:
            return state

        try:
            from evaluations.evaluators import run_all_evaluations
            from langsmith import Client
            from langsmith.run_helpers import get_current_run_tree

            question = state["question"]
            answer = state["generation"]
            retrieved_docs = state.get("retrieved_documents", [])  # Full Document objects
            retrieved_code = state.get("retrieved_code", [])  # Just the strings
            code_files = state.get("code_files", [])

            # Build context string from retrieved docs
            if retrieved_docs:
                context_parts = []
                for i, doc in enumerate(retrieved_docs, 1):
                    file_path = doc.metadata.get('source', 'unknown')
                    content = doc.page_content
                    context_parts.append(f"File: {file_path}\n{content}")
                context = "\n\n---\n\n".join(context_parts)
            elif retrieved_code and code_files:
                # Fallback: build context from strings if Document objects not available
                context_parts = []
                for code, filepath in zip(retrieved_code, code_files):
                    context_parts.append(f"File: {filepath}\n{code}")
                context = "\n\n---\n\n".join(context_parts)
            else:
                context = "No context retrieved"

            # Run all evaluations
            eval_results = run_all_evaluations(
                question=question,
                answer=answer,
                context=context,
                retrieved_documents=retrieved_docs if retrieved_docs else []
            )

            # Store evaluation results in state
            state["evaluation_results"] = eval_results

            # Send evaluation scores to LangSmith as feedback
            try:
                from langsmith import Client
                import langsmith

                client = Client()

                # Get the current run context - try multiple methods
                run_id = None

                # Method 1: Check langsmith context
                try:
                    current_run = langsmith.get_current_run_tree()
                    if current_run:
                        run_id = str(current_run.id)
                        state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                            f"🔍 Found run ID via run_tree: {run_id[:8]}..."
                        ]
                except Exception as e:
                    state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                        f"⚠️ Method 1 failed: {str(e)}"
                    ]

                # Method 2: Try to get from project runs (last run)
                if not run_id:
                    try:
                        from config.settings import LANGSMITH_PROJECT
                        recent_runs = list(client.list_runs(
                            project_name=LANGSMITH_PROJECT,
                            limit=1
                        ))
                        if recent_runs:
                            run_id = str(recent_runs[0].id)
                            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                                f"🔍 Using most recent run ID: {run_id[:8]}..."
                            ]
                    except Exception as e:
                        state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                            f"⚠️ Method 2 failed: {str(e)}"
                        ]

                if run_id:
                    # Send each metric as feedback
                    feedback_count = 0

                    accuracy = eval_results.get('accuracy', {})
                    if accuracy.get('score') is not None:
                        client.create_feedback(
                            run_id=run_id,
                            key="accuracy",
                            score=accuracy.get('score'),
                            comment=accuracy.get('reasoning', '')
                        )
                        feedback_count += 1

                    groundedness = eval_results.get('groundedness', {})
                    if groundedness.get('score') is not None:
                        client.create_feedback(
                            run_id=run_id,
                            key="groundedness",
                            score=groundedness.get('score'),
                            comment=f"Hallucinations: {groundedness.get('hallucinations', 'None')}"
                        )
                        feedback_count += 1

                    relevancy = eval_results.get('relevancy', {})
                    if relevancy.get('score') is not None:
                        client.create_feedback(
                            run_id=run_id,
                            key="retrieval_relevancy",
                            score=relevancy.get('score'),
                            comment=f"{relevancy.get('relevant_count', 0)}/{relevancy.get('total_count', 0)} chunks relevant"
                        )
                        feedback_count += 1

                    precision = eval_results.get('precision', {})
                    if precision.get('score') is not None:
                        client.create_feedback(
                            run_id=run_id,
                            key="context_precision",
                            score=precision.get('score'),
                            comment=precision.get('reasoning', '')
                        )
                        feedback_count += 1

                    state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                        f"✅ Sent {feedback_count} evaluations to LangSmith"
                    ]
                else:
                    state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                        "⚠️ Could not find run ID - feedback not sent"
                    ]

            except Exception as feedback_error:
                # Log but don't fail if feedback fails
                state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                    f"⚠️ LangSmith feedback failed: {str(feedback_error)}"
                ]

            # Add evaluation summary to intermediate steps
            summary = eval_results.get('overall', {}).get('summary', 'Evaluation completed')
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                f"📊 Evaluation:\n{summary}"
            ]

        except Exception as e:
            # Don't fail the workflow if evaluation fails
            state["intermediate_steps"] = state.get("intermediate_steps", []) + [
                f"⚠️ Evaluation skipped: {str(e)}"
            ]

        return state

    def finalize_node(self, state: GraphState) -> GraphState:
        """Finalize the answer"""
        state["final_answer"] = state["generation"]
        state["intermediate_steps"] = state.get("intermediate_steps", []) + [
            "✅ Answer finalized"
        ]
        return state
