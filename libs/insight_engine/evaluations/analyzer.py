"""
Evaluation results analyzer.

Analyzes batch evaluation results and generates comparison reports.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class EvaluationAnalyzer:
    """Analyze and compare evaluation results"""

    def __init__(self, results_dir: str = "evaluation_results"):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = results_dir

    def load_result(self, filepath: str) -> Dict[str, Any]:
        """Load a result file"""
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all evaluation results sorted by timestamp"""
        results = []

        if not os.path.exists(self.results_dir):
            return results

        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                result = self.load_result(filepath)
                result['filename'] = filename
                results.append(result)

        # Sort by execution start time
        results.sort(key=lambda x: x.get('execution', {}).get('start_time', ''), reverse=True)
        return results

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get the most recent evaluation result"""
        results = self.get_all_results()
        return results[0] if results else None

    def compare_with_baseline(self, current: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current results with baseline.

        Args:
            current: Current evaluation results
            baseline: Baseline evaluation results

        Returns:
            Comparison report
        """
        current_summary = current.get('summary', {})
        baseline_summary = baseline.get('summary', {})

        current_metrics = current_summary.get('average_metrics', {})
        baseline_metrics = baseline_summary.get('average_metrics', {})

        # Calculate deltas
        accuracy_delta = current_metrics.get('accuracy', 0) - baseline_metrics.get('accuracy', 0)
        groundedness_delta = current_metrics.get('groundedness', 0) - baseline_metrics.get('groundedness', 0)
        relevancy_delta = current_metrics.get('retrieval_relevancy', 0) - baseline_metrics.get('retrieval_relevancy', 0)
        precision_delta = current_metrics.get('context_precision', 0) - baseline_metrics.get('context_precision', 0)

        pass_rate_delta = current_summary.get('pass_rate_percentage', 0) - baseline_summary.get('pass_rate_percentage', 0)

        # Detect regressions
        regressions = []
        improvements = []

        if accuracy_delta < -0.10:
            regressions.append(f"Accuracy dropped {abs(accuracy_delta):.1%} (from {baseline_metrics.get('accuracy', 0):.2f} to {current_metrics.get('accuracy', 0):.2f})")
        elif accuracy_delta > 0.05:
            improvements.append(f"Accuracy improved {accuracy_delta:.1%}")

        if groundedness_delta < -0.15:
            regressions.append(f"Groundedness dropped {abs(groundedness_delta):.1%} (from {baseline_metrics.get('groundedness', 0):.2f} to {current_metrics.get('groundedness', 0):.2f})")
        elif groundedness_delta > 0.05:
            improvements.append(f"Groundedness improved {groundedness_delta:.1%}")

        if relevancy_delta < -0.10:
            regressions.append(f"Relevancy dropped {abs(relevancy_delta):.1%}")
        elif relevancy_delta > 0.05:
            improvements.append(f"Relevancy improved {relevancy_delta:.1%}")

        if pass_rate_delta < -10:
            regressions.append(f"Pass rate dropped {abs(pass_rate_delta):.1f}% (from {baseline_summary.get('pass_rate_percentage', 0):.1f}% to {current_summary.get('pass_rate_percentage', 0):.1f}%)")
        elif pass_rate_delta > 5:
            improvements.append(f"Pass rate improved {pass_rate_delta:.1f}%")

        return {
            "baseline_date": baseline.get('execution', {}).get('start_time', 'Unknown'),
            "current_date": current.get('execution', {}).get('start_time', 'Unknown'),
            "metric_deltas": {
                "accuracy": round(accuracy_delta, 3),
                "groundedness": round(groundedness_delta, 3),
                "retrieval_relevancy": round(relevancy_delta, 3),
                "context_precision": round(precision_delta, 3),
                "pass_rate": round(pass_rate_delta, 1)
            },
            "regressions": regressions,
            "improvements": improvements,
            "overall_status": "regression" if regressions else ("improvement" if improvements else "stable")
        }

    def generate_report(self, result: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a formatted text report.

        Args:
            result: Evaluation result to report on
            baseline: Optional baseline for comparison

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("EVALUATION RESULTS REPORT")
        lines.append("=" * 70)
        lines.append("")

        # Execution info
        execution = result.get('execution', {})
        lines.append(f"Execution Time: {execution.get('start_time', 'Unknown')}")
        lines.append(f"Duration: {execution.get('total_duration', 0):.1f}s")
        lines.append("")

        # Summary
        summary = result.get('summary', {})
        lines.append("SUMMARY")
        lines.append("-" * 70)
        lines.append(f"Total Tests: {summary.get('total_tests', 0)}")
        lines.append(f"Passed: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)} ({summary.get('pass_rate_percentage', 0):.1f}%)")
        lines.append(f"Failed Executions: {summary.get('failed_executions', 0)}")
        lines.append("")

        # Metrics
        metrics = summary.get('average_metrics', {})
        lines.append("AVERAGE METRICS")
        lines.append("-" * 70)
        lines.append(f"  Accuracy:           {metrics.get('accuracy', 0):.3f}")
        lines.append(f"  Groundedness:       {metrics.get('groundedness', 0):.3f}")
        lines.append(f"  Retrieval Relevancy: {metrics.get('retrieval_relevancy', 0):.3f}")
        lines.append(f"  Context Precision:  {metrics.get('context_precision', 0):.3f}")
        lines.append("")

        # Performance
        perf = summary.get('performance', {})
        lines.append("PERFORMANCE")
        lines.append("-" * 70)
        lines.append(f"  Total Time: {perf.get('total_time_seconds', 0):.1f}s")
        lines.append(f"  Avg per Query: {perf.get('average_time_per_query', 0):.1f}s")
        lines.append(f"  Estimated Cost: ${perf.get('estimated_cost_usd', 0):.3f}")
        lines.append("")

        # By category
        by_category = summary.get('by_category', {})
        if by_category:
            lines.append("BY CATEGORY")
            lines.append("-" * 70)
            for category, stats in by_category.items():
                lines.append(f"  {category.upper()}")
                lines.append(f"    Passed: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.1f}%)")
                lines.append(f"    Avg Accuracy: {stats['avg_accuracy']:.3f}")
            lines.append("")

        # Individual test results
        lines.append("TEST DETAILS")
        lines.append("-" * 70)
        for test_result in result.get('results', []):
            test_id = test_result.get('test_id', 'Unknown')
            query = test_result.get('query', '')
            passed = test_result.get('pass_status', {}).get('overall_pass', False)
            status_icon = "✅" if passed else "❌"
            accuracy = test_result.get('evaluation_metrics', {}).get('accuracy', 0)

            lines.append(f"  [{test_id}] {status_icon} {query[:50]}{'...' if len(query) > 50 else ''}")
            lines.append(f"      Accuracy: {accuracy:.2f}")

            if not test_result.get('success'):
                lines.append(f"      ERROR: {test_result.get('error', 'Unknown')}")

        lines.append("")

        # Comparison with baseline
        if baseline:
            comparison = self.compare_with_baseline(result, baseline)
            lines.append("COMPARISON WITH BASELINE")
            lines.append("-" * 70)
            lines.append(f"Baseline Date: {comparison['baseline_date']}")
            lines.append("")

            deltas = comparison['metric_deltas']
            lines.append("Metric Changes:")
            for metric, delta in deltas.items():
                direction = "↑" if delta > 0 else ("↓" if delta < 0 else "↔")
                lines.append(f"  {metric}: {delta:+.3f} {direction}")
            lines.append("")

            if comparison['regressions']:
                lines.append("⚠️  REGRESSIONS DETECTED:")
                for reg in comparison['regressions']:
                    lines.append(f"  - {reg}")
                lines.append("")

            if comparison['improvements']:
                lines.append("✅ IMPROVEMENTS:")
                for imp in comparison['improvements']:
                    lines.append(f"  - {imp}")
                lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def print_summary(self, result: Dict[str, Any]):
        """Print a quick summary to console"""
        summary = result.get('summary', {})
        metrics = summary.get('average_metrics', {})
        perf = summary.get('performance', {})

        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"\nPass Rate: {summary.get('pass_rate_percentage', 0):.1f}% ({summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)})")
        print(f"\nAverage Metrics:")
        print(f"  - Accuracy:    {metrics.get('accuracy', 0):.3f}")
        print(f"  - Groundedness: {metrics.get('groundedness', 0):.3f}")
        print(f"  - Relevancy:   {metrics.get('retrieval_relevancy', 0):.3f}")
        print(f"  - Precision:   {metrics.get('context_precision', 0):.3f}")
        print(f"\nPerformance:")
        print(f"  - Total Time: {perf.get('total_time_seconds', 0):.1f}s")
        print(f"  - Cost: ${perf.get('estimated_cost_usd', 0):.3f}")
        print("=" * 70 + "\n")
