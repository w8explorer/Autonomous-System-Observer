"""
Batch evaluation runner for systematic RAG testing.

This script runs a dataset of test queries through the RAG pipeline
and collects evaluation metrics for quality tracking.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path


class BatchEvaluationRunner:
    """Run batch evaluations on a test dataset"""

    def __init__(self, dataset_path: str, java_code_path: str):
        """
        Initialize batch runner.

        Args:
            dataset_path: Path to evaluation dataset JSON
            java_code_path: Path to Java code directory
        """
        self.dataset_path = dataset_path
        self.java_code_path = java_code_path
        self.results = []
        self.start_time = None
        self.end_time = None

    def load_dataset(self) -> Dict[str, Any]:
        """Load evaluation dataset from JSON file"""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    def initialize_system(self):
        """Initialize RAG system (vectorstore, LLM, workflow)"""
        from core.initializer import initialize_system

        print(f"📚 Initializing system with Java code from: {self.java_code_path}")
        app, retriever, files_count = initialize_system(self.java_code_path)

        if not app:
            raise RuntimeError("Failed to initialize RAG system")

        print(f"✅ System initialized with {files_count} files\n")
        return app

    def run_single_test(self, app, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single test case through the RAG pipeline.

        Args:
            app: Compiled LangGraph application
            test_case: Test case dictionary

        Returns:
            Test result with metrics
        """
        query = test_case['query']
        test_id = test_case['id']
        category = test_case['category']

        print(f"  [{test_id}] Running: {query[:50]}{'...' if len(query) > 50 else ''}")

        start_time = time.time()

        # Create initial state
        initial_state = {
            "question": query,
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

        try:
            # Run through RAG pipeline
            result = app.invoke(initial_state)

            # Extract evaluation results
            eval_results = result.get("evaluation_results", {})

            execution_time = time.time() - start_time

            # Build result object
            test_result = {
                "test_id": test_id,
                "query": query,
                "category": category,
                "difficulty": test_case.get('difficulty', 'unknown'),
                "timestamp": datetime.now().isoformat(),
                "execution_time": round(execution_time, 2),
                "answer": result.get("final_answer", ""),
                "retrieved_files": result.get("code_files", []),
                "retrieved_chunks_count": len(result.get("retrieved_code", [])),
                "evaluation_metrics": {
                    "accuracy": eval_results.get('accuracy', {}).get('score', 0.0),
                    "groundedness": eval_results.get('groundedness', {}).get('score', 0.0),
                    "retrieval_relevancy": eval_results.get('relevancy', {}).get('score', 0.0),
                    "context_precision": eval_results.get('precision', {}).get('score', 0.0),
                    "relevancy_percentage": eval_results.get('relevancy', {}).get('percentage', 0.0)
                },
                "pass_status": {
                    "accuracy_pass": eval_results.get('accuracy', {}).get('pass', False),
                    "groundedness_pass": eval_results.get('groundedness', {}).get('pass', False),
                    "relevancy_pass": eval_results.get('relevancy', {}).get('pass', False),
                    "precision_pass": eval_results.get('precision', {}).get('pass', False),
                    "overall_pass": eval_results.get('overall', {}).get('all_passed', False)
                },
                "details": {
                    "hallucinations": eval_results.get('groundedness', {}).get('hallucinations', 'Unknown'),
                    "accuracy_reasoning": eval_results.get('accuracy', {}).get('reasoning', ''),
                    "intermediate_steps_count": len(result.get("intermediate_steps", []))
                },
                "expected_behavior": test_case.get('expected_behavior', {}),
                "success": True,
                "error": None
            }

            # Display result
            status = "✅ PASS" if test_result["pass_status"]["overall_pass"] else "⚠️  PARTIAL"
            accuracy = test_result["evaluation_metrics"]["accuracy"]
            print(f"      {status} (accuracy: {accuracy:.2f}, time: {execution_time:.1f}s)")

            return test_result

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"      ❌ ERROR: {str(e)}")

            return {
                "test_id": test_id,
                "query": query,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "execution_time": round(execution_time, 2),
                "success": False,
                "error": str(e),
                "evaluation_metrics": {
                    "accuracy": 0.0,
                    "groundedness": 0.0,
                    "retrieval_relevancy": 0.0,
                    "context_precision": 0.0
                },
                "pass_status": {
                    "overall_pass": False
                }
            }

    def run_batch(self) -> Dict[str, Any]:
        """
        Run all test cases in the dataset.

        Returns:
            Batch results with summary statistics
        """
        self.start_time = time.time()

        # Load dataset
        print("📊 Loading evaluation dataset...")
        dataset = self.load_dataset()
        test_cases = dataset.get('test_cases', [])
        metadata = dataset.get('metadata', {})

        print(f"✅ Loaded {len(test_cases)} test cases\n")

        # Initialize system
        app = self.initialize_system()

        # Run tests
        print(f"🚀 Running batch evaluation...\n")

        for i, test_case in enumerate(test_cases, 1):
            print(f"[{i}/{len(test_cases)}]")
            result = self.run_single_test(app, test_case)
            self.results.append(result)
            print()  # Blank line between tests

        self.end_time = time.time()

        # Generate summary
        summary = self._generate_summary(metadata)

        return {
            "metadata": metadata,
            "execution": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "total_duration": round(self.end_time - self.start_time, 2)
            },
            "results": self.results,
            "summary": summary
        }

    def _generate_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics from results"""
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.get('success', False)])

        # Calculate pass rates
        passed_tests = len([r for r in self.results if r.get('pass_status', {}).get('overall_pass', False)])
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        # Calculate average metrics
        accuracy_scores = [r['evaluation_metrics']['accuracy'] for r in self.results if r.get('success')]
        groundedness_scores = [r['evaluation_metrics']['groundedness'] for r in self.results if r.get('success')]
        relevancy_scores = [r['evaluation_metrics']['retrieval_relevancy'] for r in self.results if r.get('success')]
        precision_scores = [r['evaluation_metrics']['context_precision'] for r in self.results if r.get('success')]

        avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
        avg_groundedness = sum(groundedness_scores) / len(groundedness_scores) if groundedness_scores else 0
        avg_relevancy = sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0

        # Calculate total time and cost
        total_time = sum([r.get('execution_time', 0) for r in self.results])
        estimated_cost = total_tests * 0.013  # $0.013 per query (base + evaluation)

        return {
            "total_tests": total_tests,
            "successful_executions": successful_tests,
            "failed_executions": total_tests - successful_tests,
            "passed_tests": passed_tests,
            "pass_rate_percentage": round(pass_rate, 1),
            "average_metrics": {
                "accuracy": round(avg_accuracy, 3),
                "groundedness": round(avg_groundedness, 3),
                "retrieval_relevancy": round(avg_relevancy, 3),
                "context_precision": round(avg_precision, 3)
            },
            "performance": {
                "total_time_seconds": round(total_time, 2),
                "average_time_per_query": round(total_time / total_tests, 2) if total_tests > 0 else 0,
                "estimated_cost_usd": round(estimated_cost, 3)
            },
            "by_category": self._summarize_by_category()
        }

    def _summarize_by_category(self) -> Dict[str, Any]:
        """Summarize results by test category"""
        categories = {}

        for result in self.results:
            category = result.get('category', 'unknown')

            if category not in categories:
                categories[category] = {
                    'total': 0,
                    'passed': 0,
                    'avg_accuracy': []
                }

            categories[category]['total'] += 1
            if result.get('pass_status', {}).get('overall_pass', False):
                categories[category]['passed'] += 1
            if result.get('success'):
                categories[category]['avg_accuracy'].append(result['evaluation_metrics']['accuracy'])

        # Calculate averages
        for category, stats in categories.items():
            if stats['avg_accuracy']:
                stats['avg_accuracy'] = round(sum(stats['avg_accuracy']) / len(stats['avg_accuracy']), 3)
            else:
                stats['avg_accuracy'] = 0.0
            stats['pass_rate'] = round((stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0, 1)

        return categories

    def save_results(self, output_dir: str = "evaluation_results") -> str:
        """
        Save results to JSON file.

        Args:
            output_dir: Directory to save results

        Returns:
            Path to saved file
        """
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_mini_evaluation.json"
        filepath = os.path.join(output_dir, filename)

        # Save results
        results_data = {
            "metadata": self.load_dataset().get('metadata', {}),
            "execution": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
                "total_duration": round(self.end_time - self.start_time, 2) if self.end_time and self.start_time else 0
            },
            "results": self.results,
            "summary": self._generate_summary({})
        }

        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)

        return filepath
