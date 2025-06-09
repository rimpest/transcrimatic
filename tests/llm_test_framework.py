"""
Comprehensive testing framework for LLM Analyzer with local Ollama models.
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm import LLMAnalyzer, Conversation, Analysis
from tests.test_data.spanish_conversations import (
    CONVERSATIONS, EDGE_CASES, get_all_conversations,
    get_conversation_by_type, create_whisper_style_output
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of a single test."""
    test_name: str
    conversation_id: str
    passed: bool
    duration: float
    provider_used: str
    errors: List[str]
    metrics: Dict[str, any]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    timestamp: datetime
    results: List[TestResult]
    total_duration: float
    
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)
    
    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return (self.passed_count / len(self.results)) * 100
    
    def to_dict(self):
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "results": [r.to_dict() for r in self.results],
            "total_duration": self.total_duration,
            "summary": {
                "total_tests": len(self.results),
                "passed": self.passed_count,
                "failed": self.failed_count,
                "success_rate": f"{self.success_rate:.1f}%"
            }
        }


class LLMTestFramework:
    """Comprehensive testing framework for LLM Analyzer."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize test framework."""
        self.config = config or self._get_default_config()
        self.analyzer = self._setup_analyzer()
        self.results = []
        
    def _get_default_config(self) -> Dict:
        """Get default configuration for testing."""
        return {
            "llm": {
                "provider": "ollama",
                "fallback_order": ["ollama"],
                "ollama": {
                    "enabled": True,
                    "host": "localhost",
                    "port": 11434,
                    "model": "llama3.2",  # Updated to use llama3.2
                    "timeout": 120,
                    "temperature": 0.3
                },
                "cost_optimization": {
                    "cache_similar_prompts": True,
                    "batch_conversations": True,
                    "max_batch_size": 3
                }
            }
        }
    
    def _setup_analyzer(self) -> LLMAnalyzer:
        """Setup LLM Analyzer with test configuration."""
        class MockConfig:
            def get(self, key, default=None):
                return self.config.get(key, default)
        
        mock_config = MockConfig()
        mock_config.config = self.config
        return LLMAnalyzer(mock_config)
    
    def test_single_conversation(self, conv_data: Dict) -> TestResult:
        """Test analysis of a single conversation."""
        test_name = f"test_conversation_{conv_data.get('id', 'unknown')}"
        logger.info(f"Running {test_name}")
        
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Create conversation object
            conversation = Conversation(
                id=conv_data['id'],
                transcript=conv_data.get('raw_transcript', ''),
                speaker_transcript=conv_data.get('speaker_transcript', ''),
                speakers=conv_data.get('speakers', []),
                duration=conv_data.get('duration', 0),
                timestamp=datetime.now()
            )
            
            # Analyze conversation
            analysis = self.analyzer.analyze_conversation(conversation)
            
            # Validate results
            errors.extend(self._validate_analysis(analysis, conv_data))
            
            # Calculate metrics
            metrics = self._calculate_metrics(analysis, conv_data)
            
            # Test passed if no errors
            passed = len(errors) == 0
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=test_name,
                conversation_id=conv_data['id'],
                passed=passed,
                duration=duration,
                provider_used=analysis.llm_provider,
                errors=errors,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return TestResult(
                test_name=test_name,
                conversation_id=conv_data.get('id', 'unknown'),
                passed=False,
                duration=time.time() - start_time,
                provider_used='unknown',
                errors=[f"Exception: {str(e)}"],
                metrics={}
            )
    
    def _validate_analysis(self, analysis: Analysis, expected: Dict) -> List[str]:
        """Validate analysis results against expected outcomes."""
        errors = []
        
        # Check if summary exists and is reasonable
        if not analysis.summary:
            errors.append("No summary generated")
        elif len(analysis.summary) < 50:
            errors.append("Summary too short")
        
        # Check key points
        if len(analysis.key_points) == 0:
            errors.append("No key points extracted")
        
        # Validate tasks if expected
        if 'expected_tasks' in expected:
            expected_tasks = expected['expected_tasks']
            if len(analysis.tasks) == 0 and len(expected_tasks) > 0:
                errors.append(f"Expected {len(expected_tasks)} tasks, found 0")
            
            # Check task quality
            for task in analysis.tasks:
                if not task.description:
                    errors.append("Task with empty description")
                if task.priority not in ['alta', 'media', 'baja']:
                    errors.append(f"Invalid task priority: {task.priority}")
        
        # Check participants
        if len(analysis.participants) != len(expected.get('speakers', [])):
            errors.append(f"Participant count mismatch: {len(analysis.participants)} vs {len(expected.get('speakers', []))}")
        
        return errors
    
    def _calculate_metrics(self, analysis: Analysis, expected: Dict) -> Dict:
        """Calculate quality metrics for the analysis."""
        metrics = {
            "summary_length": len(analysis.summary),
            "key_points_count": len(analysis.key_points),
            "tasks_found": len(analysis.tasks),
            "todos_found": len(analysis.todos),
            "followups_found": len(analysis.followups),
            "participants_detected": len(analysis.participants)
        }
        
        # Task extraction accuracy if expected tasks provided
        if 'expected_tasks' in expected:
            expected_count = len(expected['expected_tasks'])
            found_count = len(analysis.tasks)
            if expected_count > 0:
                metrics['task_recall'] = min(found_count / expected_count, 1.0)
            else:
                metrics['task_recall'] = 1.0 if found_count == 0 else 0.0
        
        # Average words per key point
        if analysis.key_points:
            total_words = sum(len(point.split()) for point in analysis.key_points)
            metrics['avg_words_per_keypoint'] = total_words / len(analysis.key_points)
        
        return metrics
    
    def test_batch_processing(self, conversations: List[Dict]) -> TestResult:
        """Test batch processing capabilities."""
        test_name = "test_batch_processing"
        logger.info(f"Running {test_name} with {len(conversations)} conversations")
        
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Create conversation objects
            conv_objects = []
            for conv_data in conversations:
                conv_objects.append(Conversation(
                    id=conv_data['id'],
                    transcript=conv_data.get('raw_transcript', ''),
                    speaker_transcript=conv_data.get('speaker_transcript', ''),
                    speakers=conv_data.get('speakers', []),
                    duration=conv_data.get('duration', 0),
                    timestamp=datetime.now()
                ))
            
            # Batch analyze
            analyses = self.analyzer.batch_analyze(conv_objects)
            
            # Validate batch results
            if len(analyses) != len(conversations):
                errors.append(f"Batch size mismatch: {len(analyses)} vs {len(conversations)}")
            
            # Calculate batch metrics
            metrics = {
                "batch_size": len(conversations),
                "successful_analyses": len(analyses),
                "total_duration": time.time() - start_time,
                "avg_time_per_conversation": (time.time() - start_time) / len(conversations)
            }
            
            passed = len(errors) == 0
            
            return TestResult(
                test_name=test_name,
                conversation_id="batch",
                passed=passed,
                duration=time.time() - start_time,
                provider_used=analyses[0].llm_provider if analyses else "unknown",
                errors=errors,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return TestResult(
                test_name=test_name,
                conversation_id="batch",
                passed=False,
                duration=time.time() - start_time,
                provider_used="unknown",
                errors=[f"Exception: {str(e)}"],
                metrics={}
            )
    
    def test_edge_cases(self) -> List[TestResult]:
        """Test edge cases and unusual inputs."""
        results = []
        
        for case_name, case_data in EDGE_CASES.items():
            test_name = f"test_edge_case_{case_name}"
            logger.info(f"Running {test_name}")
            
            # Convert edge case to full conversation format
            conv_data = {
                "id": f"edge_{case_name}",
                "title": f"Edge Case: {case_name}",
                "raw_transcript": case_data.get('speaker_transcript', ''),
                "speaker_transcript": case_data['speaker_transcript'],
                "duration": case_data['duration'],
                "speakers": case_data['speakers']
            }
            
            result = self.test_single_conversation(conv_data)
            result.test_name = test_name
            results.append(result)
        
        return results
    
    def test_performance(self, num_conversations: int = 10) -> TestResult:
        """Test performance with multiple conversations."""
        test_name = f"test_performance_{num_conversations}_convs"
        logger.info(f"Running {test_name}")
        
        # Use sample conversations, repeat if necessary
        all_convs = get_all_conversations()
        test_convs = []
        for i in range(num_conversations):
            test_convs.append(all_convs[i % len(all_convs)])
        
        # Test batch processing performance
        return self.test_batch_processing(test_convs)
    
    def test_daily_summary(self) -> TestResult:
        """Test daily summary generation."""
        test_name = "test_daily_summary"
        logger.info(f"Running {test_name}")
        
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Analyze all conversations
            all_convs = get_all_conversations()
            conv_objects = []
            
            for conv_data in all_convs:
                conv_objects.append(Conversation(
                    id=conv_data['id'],
                    transcript=conv_data.get('raw_transcript', ''),
                    speaker_transcript=conv_data.get('speaker_transcript', ''),
                    speakers=conv_data.get('speakers', []),
                    duration=conv_data.get('duration', 0),
                    timestamp=datetime.now()
                ))
            
            # Get analyses
            analyses = self.analyzer.batch_analyze(conv_objects)
            
            # Generate daily summary
            daily_summary = self.analyzer.generate_daily_summary(analyses)
            
            # Validate daily summary
            if not daily_summary.highlights:
                errors.append("No highlights generated")
            
            if daily_summary.total_conversations != len(analyses):
                errors.append("Conversation count mismatch in daily summary")
            
            # Calculate metrics
            metrics = {
                "total_conversations": daily_summary.total_conversations,
                "total_tasks": len(daily_summary.all_tasks),
                "total_todos": len(daily_summary.all_todos),
                "total_followups": len(daily_summary.all_followups),
                "highlights_count": len(daily_summary.highlights),
                "unique_speakers": len(daily_summary.speaker_participation)
            }
            
            passed = len(errors) == 0
            
            return TestResult(
                test_name=test_name,
                conversation_id="daily_summary",
                passed=passed,
                duration=time.time() - start_time,
                provider_used="ollama",
                errors=errors,
                metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {str(e)}")
            return TestResult(
                test_name=test_name,
                conversation_id="daily_summary",
                passed=False,
                duration=time.time() - start_time,
                provider_used="unknown",
                errors=[f"Exception: {str(e)}"],
                metrics={}
            )
    
    def run_all_tests(self) -> TestSuite:
        """Run all tests and return results."""
        suite_start = time.time()
        results = []
        
        logger.info("Starting comprehensive LLM test suite")
        
        # Test 1: Individual conversations
        logger.info("\n=== Testing Individual Conversations ===")
        for conv_type, conv_data in CONVERSATIONS.items():
            result = self.test_single_conversation(conv_data)
            results.append(result)
            self._print_result(result)
        
        # Test 2: Edge cases
        logger.info("\n=== Testing Edge Cases ===")
        edge_results = self.test_edge_cases()
        results.extend(edge_results)
        for result in edge_results:
            self._print_result(result)
        
        # Test 3: Batch processing
        logger.info("\n=== Testing Batch Processing ===")
        batch_result = self.test_batch_processing(get_all_conversations()[:3])
        results.append(batch_result)
        self._print_result(batch_result)
        
        # Test 4: Performance
        logger.info("\n=== Testing Performance ===")
        perf_result = self.test_performance(5)
        results.append(perf_result)
        self._print_result(perf_result)
        
        # Test 5: Daily summary
        logger.info("\n=== Testing Daily Summary ===")
        summary_result = self.test_daily_summary()
        results.append(summary_result)
        self._print_result(summary_result)
        
        suite = TestSuite(
            name="LLM Analyzer Comprehensive Test Suite",
            timestamp=datetime.now(),
            results=results,
            total_duration=time.time() - suite_start
        )
        
        return suite
    
    def _print_result(self, result: TestResult):
        """Print test result to console."""
        status = "✓ PASSED" if result.passed else "✗ FAILED"
        print(f"{status} - {result.test_name} ({result.duration:.2f}s)")
        
        if not result.passed:
            for error in result.errors:
                print(f"  - Error: {error}")
        
        if result.metrics:
            print(f"  - Metrics: {json.dumps(result.metrics, indent=2)}")
    
    def save_results(self, suite: TestSuite, filename: str):
        """Save test results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(suite.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def generate_report(self, suite: TestSuite) -> str:
        """Generate a markdown report of test results."""
        report = f"""# LLM Analyzer Test Report

**Date**: {suite.timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**Total Duration**: {suite.total_duration:.2f} seconds

## Summary

- **Total Tests**: {len(suite.results)}
- **Passed**: {suite.passed_count}
- **Failed**: {suite.failed_count}
- **Success Rate**: {suite.success_rate:.1f}%

## Test Results

"""
        
        # Group results by test type
        individual_tests = [r for r in suite.results if r.test_name.startswith('test_conversation_')]
        edge_tests = [r for r in suite.results if r.test_name.startswith('test_edge_case_')]
        other_tests = [r for r in suite.results if not r.test_name.startswith('test_conversation_') 
                       and not r.test_name.startswith('test_edge_case_')]
        
        # Individual conversation tests
        if individual_tests:
            report += "### Individual Conversation Tests\n\n"
            report += "| Test | Status | Duration | Provider | Key Metrics |\n"
            report += "|------|--------|----------|----------|-------------|\n"
            
            for result in individual_tests:
                status = "✓" if result.passed else "✗"
                key_metrics = f"Tasks: {result.metrics.get('tasks_found', 0)}, Todos: {result.metrics.get('todos_found', 0)}"
                report += f"| {result.conversation_id} | {status} | {result.duration:.2f}s | {result.provider_used} | {key_metrics} |\n"
        
        # Edge case tests
        if edge_tests:
            report += "\n### Edge Case Tests\n\n"
            report += "| Test | Status | Duration | Errors |\n"
            report += "|------|--------|----------|--------|\n"
            
            for result in edge_tests:
                status = "✓" if result.passed else "✗"
                errors = len(result.errors) if result.errors else 0
                report += f"| {result.test_name.replace('test_edge_case_', '')} | {status} | {result.duration:.2f}s | {errors} |\n"
        
        # Other tests
        if other_tests:
            report += "\n### System Tests\n\n"
            for result in other_tests:
                status = "PASSED" if result.passed else "FAILED"
                report += f"#### {result.test_name}\n\n"
                report += f"- **Status**: {status}\n"
                report += f"- **Duration**: {result.duration:.2f}s\n"
                report += f"- **Provider**: {result.provider_used}\n"
                
                if result.metrics:
                    report += "- **Metrics**:\n"
                    for key, value in result.metrics.items():
                        report += f"  - {key}: {value}\n"
                
                if result.errors:
                    report += "- **Errors**:\n"
                    for error in result.errors:
                        report += f"  - {error}\n"
                
                report += "\n"
        
        # Failed tests details
        failed_tests = [r for r in suite.results if not r.passed]
        if failed_tests:
            report += "## Failed Tests Details\n\n"
            for result in failed_tests:
                report += f"### {result.test_name}\n\n"
                for error in result.errors:
                    report += f"- {error}\n"
                report += "\n"
        
        return report


def main():
    """Main entry point for testing."""
    parser = argparse.ArgumentParser(description='Test LLM Analyzer with local Ollama')
    parser.add_argument('--test', choices=['all', 'single', 'batch', 'edge', 'performance', 'summary'],
                       default='all', help='Type of test to run')
    parser.add_argument('--conversation', help='Specific conversation type to test')
    parser.add_argument('--output', default='test_results.json', help='Output file for results')
    parser.add_argument('--report', default='test_report.md', help='Output file for markdown report')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test framework
    framework = LLMTestFramework()
    
    # Run tests based on argument
    if args.test == 'single' and args.conversation:
        conv_data = get_conversation_by_type(args.conversation)
        if conv_data:
            result = framework.test_single_conversation(conv_data)
            framework._print_result(result)
        else:
            print(f"Conversation type '{args.conversation}' not found")
            print(f"Available types: {', '.join(CONVERSATIONS.keys())}")
    
    elif args.test == 'edge':
        results = framework.test_edge_cases()
        for result in results:
            framework._print_result(result)
    
    elif args.test == 'batch':
        result = framework.test_batch_processing(get_all_conversations()[:3])
        framework._print_result(result)
    
    elif args.test == 'performance':
        result = framework.test_performance(10)
        framework._print_result(result)
    
    elif args.test == 'summary':
        result = framework.test_daily_summary()
        framework._print_result(result)
    
    else:  # all
        suite = framework.run_all_tests()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SUITE SUMMARY")
        print("="*60)
        print(f"Total Tests: {len(suite.results)}")
        print(f"Passed: {suite.passed_count}")
        print(f"Failed: {suite.failed_count}")
        print(f"Success Rate: {suite.success_rate:.1f}%")
        print(f"Total Duration: {suite.total_duration:.2f} seconds")
        
        # Save results
        framework.save_results(suite, args.output)
        
        # Generate and save report
        report = framework.generate_report(suite)
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()