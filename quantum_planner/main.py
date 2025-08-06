"""Main entry point for Quantum-Inspired Task Planner."""

import sys
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import click
from datetime import datetime, timedelta

# Add parent directory to path for microdiff integration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .core.scheduler import QuantumInspiredScheduler, Resource
from .core.task import Task, TaskPriority, TaskStatus
from .utils.logging_config import setup_logging, LogContext
from .utils.error_handling import handle_errors, get_error_handler
from .utils.performance import PerformanceMonitor
from .utils.scaling import ScalingOrchestrator, ScalingConfig
from .utils.validation import TaskValidator, ResourceValidator
from .utils.visualization import ScheduleVisualizer

# Setup logging
setup_logging(log_level="INFO", log_dir="logs/quantum_planner")
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="1.0.0")
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--log-level', default='INFO', help='Logging level')
@click.option('--performance-monitoring', is_flag=True, help='Enable performance monitoring')
@click.option('--scaling', is_flag=True, help='Enable advanced scaling features')
@click.pass_context
def main(ctx, config, log_level, performance_monitoring, scaling):
    """Quantum-Inspired Task Planner: Advanced scheduling using quantum principles.
    
    This CLI provides quantum-inspired task scheduling capabilities with
    advanced optimization, scaling, and monitoring features.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    ctx.obj['config'] = {}
    if config:
        try:
            with open(config, 'r') as f:
                ctx.obj['config'] = json.load(f)
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)
    
    # Setup logging level
    setup_logging(log_level=log_level)
    ctx.obj['log_level'] = log_level
    
    # Initialize performance monitoring
    if performance_monitoring:
        ctx.obj['performance_monitor'] = PerformanceMonitor()
        ctx.obj['performance_monitor'].start_monitoring()
    else:
        ctx.obj['performance_monitor'] = None
    
    # Initialize scaling
    if scaling:
        scaling_config = ScalingConfig()
        ctx.obj['scaling_orchestrator'] = ScalingOrchestrator(scaling_config)
    else:
        ctx.obj['scaling_orchestrator'] = None
    
    logger.info("Quantum Task Planner initialized")


@main.command()
@click.argument('tasks_file', type=click.Path(exists=True))
@click.argument('resources_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for schedule')
@click.option('--algorithm', type=click.Choice(['quantum_annealing', 'superposition', 'hybrid']),
              default='hybrid', help='Optimization algorithm to use')
@click.option('--max-time', default=300, help='Maximum optimization time in seconds')
@click.option('--visualize', is_flag=True, help='Generate schedule visualization')
@click.option('--validate', is_flag=True, help='Validate inputs before scheduling')
@click.pass_context
def schedule(ctx, tasks_file, resources_file, output, algorithm, max_time, visualize, validate):
    """Create optimal schedule from tasks and resources files.
    
    Takes JSON files containing task definitions and resource specifications,
    then generates an optimized schedule using quantum-inspired algorithms.
    """
    with LogContext(operation="schedule"):
        try:
            logger.info(f"Starting schedule optimization with {algorithm} algorithm")
            
            # Load tasks and resources
            with open(tasks_file, 'r') as f:
                tasks_data = json.load(f)
            
            with open(resources_file, 'r') as f:
                resources_data = json.load(f)
            
            # Convert to objects
            tasks = []
            for task_data in tasks_data:
                task = Task.from_dict(task_data)
                tasks.append(task)
            
            resources = {}
            for resource_data in resources_data:
                resource = Resource(
                    id=resource_data['id'],
                    name=resource_data['name'],
                    capacity=resource_data['capacity'],
                    available_capacity=resource_data.get('available_capacity', resource_data['capacity']),
                    cost_per_minute=resource_data.get('cost_per_minute', 0.0)
                )
                resources[resource.id] = resource
            
            logger.info(f"Loaded {len(tasks)} tasks and {len(resources)} resources")
            
            # Validation
            if validate:
                validator = TaskValidator(strict_mode=True)
                validation_result = validator.validate_task_list(tasks)
                
                if not validation_result.is_valid():
                    click.echo("Validation failed:")
                    for error in validation_result.errors:
                        click.echo(f"  ERROR: {error}")
                    for warning in validation_result.warnings:
                        click.echo(f"  WARNING: {warning}")
                    
                    if validation_result.errors:
                        sys.exit(1)
                
                resource_validator = ResourceValidator()
                resource_validation = resource_validator.validate_resource_list(list(resources.values()))
                
                if not resource_validation.is_valid():
                    click.echo("Resource validation failed:")
                    for error in resource_validation.errors:
                        click.echo(f"  ERROR: {error}")
                    sys.exit(1)
            
            # Initialize scheduler
            scheduler = QuantumInspiredScheduler(
                optimization_method=algorithm,
                enable_superposition=True,
                enable_entanglement=True,
                performance_monitoring=ctx.obj['performance_monitor'] is not None
            )
            
            # Add tasks and resources
            for task in tasks:
                scheduler.add_task(task)
            
            for resource in resources.values():
                scheduler.add_resource(resource)
            
            # Create schedule
            with ctx.obj['performance_monitor'].start_operation("schedule_optimization") if ctx.obj['performance_monitor'] else nullcontext():
                result = scheduler.create_optimal_schedule(optimization_time_limit=max_time)
            
            if not result.success:
                click.echo(f"Scheduling failed: {result.error_message}", err=True)
                sys.exit(1)
            
            # Display results
            click.echo("\n" + "="*60)
            click.echo("QUANTUM SCHEDULE OPTIMIZATION RESULTS")
            click.echo("="*60)
            click.echo(f"Algorithm: {algorithm}")
            click.echo(f"Tasks Scheduled: {len(result.schedule)}")
            click.echo(f"Total Completion Time: {result.total_completion_time} minutes")
            click.echo(f"Conflicts Resolved: {result.conflicts_resolved}")
            click.echo(f"Dependencies Satisfied: {result.dependencies_satisfied}")
            
            # Resource utilization
            click.echo("\nResource Utilization:")
            for resource_id, utilization in result.resource_utilization.items():
                click.echo(f"  {resource_id}: {utilization:.1%}")
            
            # Optimization metrics
            if result.optimization_metrics:
                click.echo("\nOptimization Metrics:")
                for metric, value in result.optimization_metrics.items():
                    click.echo(f"  {metric}: {value}")
            
            # Quantum metrics
            if result.quantum_metrics:
                click.echo("\nQuantum Metrics:")
                for metric, value in result.quantum_metrics.items():
                    if isinstance(value, float):
                        click.echo(f"  {metric}: {value:.4f}")
                    else:
                        click.echo(f"  {metric}: {value}")
            
            # Save results
            if output:
                output_data = {
                    'algorithm': algorithm,
                    'optimization_time_limit': max_time,
                    'schedule': result.schedule,
                    'total_completion_time': result.total_completion_time,
                    'resource_utilization': result.resource_utilization,
                    'optimization_metrics': result.optimization_metrics,
                    'quantum_metrics': result.quantum_metrics,
                    'conflicts_resolved': result.conflicts_resolved,
                    'dependencies_satisfied': result.dependencies_satisfied,
                    'success': result.success,
                    'timestamp': datetime.now().isoformat()
                }
                
                output_path = Path(output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2, default=str)
                
                click.echo(f"\nSchedule saved to {output}")
            
            # Generate visualization
            if visualize:
                viz_dir = Path(output).parent / "visualizations" if output else Path("visualizations")
                viz_dir.mkdir(exist_ok=True)
                
                visualizer = ScheduleVisualizer()
                
                # Generate Gantt chart
                gantt_file = viz_dir / "schedule_gantt.png"
                visualizer.create_gantt_chart(result.schedule, str(gantt_file))
                
                # Generate resource utilization chart
                util_file = viz_dir / "resource_utilization.png"
                visualizer.create_resource_utilization_chart(result.resource_utilization, str(util_file))
                
                # Generate dependency graph
                dep_graph = scheduler.get_task_dependencies_graph()
                dep_file = viz_dir / "dependency_graph.png"
                visualizer.create_dependency_graph(dep_graph, str(dep_file))
                
                click.echo(f"\nVisualizations saved to {viz_dir}")
            
        except Exception as e:
            logger.error(f"Schedule optimization failed: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@main.command()
@click.argument('num_tasks', type=int)
@click.option('--priorities', multiple=True, default=['HIGH', 'MEDIUM', 'LOW'],
              help='Task priorities to use')
@click.option('--resources', default=4, help='Number of resources to create')
@click.option('--output-tasks', required=True, help='Output file for tasks')
@click.option('--output-resources', required=True, help='Output file for resources')
@click.option('--complexity', type=click.Choice(['simple', 'medium', 'complex']),
              default='medium', help='Task complexity level')
@click.pass_context
def generate(ctx, num_tasks, priorities, resources, output_tasks, output_resources, complexity):
    """Generate sample tasks and resources for testing.
    
    Creates realistic task and resource datasets for testing the scheduler
    with various complexity levels and constraints.
    """
    import random
    import uuid
    
    logger.info(f"Generating {num_tasks} tasks and {resources} resources")
    
    # Generate tasks
    tasks_data = []
    priority_values = [TaskPriority[p] for p in priorities if p in TaskPriority.__members__]
    
    complexity_params = {
        'simple': {'max_duration': 120, 'max_deps': 2, 'dep_prob': 0.2, 'conflict_prob': 0.1},
        'medium': {'max_duration': 480, 'max_deps': 5, 'dep_prob': 0.3, 'conflict_prob': 0.2},
        'complex': {'max_duration': 1440, 'max_deps': 10, 'dep_prob': 0.4, 'conflict_prob': 0.3}
    }
    
    params = complexity_params[complexity]
    task_ids = [str(uuid.uuid4()) for _ in range(num_tasks)]
    
    for i in range(num_tasks):
        task_id = task_ids[i]
        
        # Basic task properties
        task_data = {
            'id': task_id,
            'name': f'Task-{i+1:03d}',
            'description': f'Generated task {i+1} with {complexity} complexity',
            'priority': random.choice(priority_values).name,
            'status': 'pending',
            'estimated_duration': random.randint(15, params['max_duration']),
        }
        
        # Add deadline (30% of tasks have deadlines)
        if random.random() < 0.3:
            deadline = datetime.now() + timedelta(
                hours=random.randint(24, 168)  # 1 day to 1 week
            )
            task_data['deadline'] = deadline.isoformat()
        
        # Add dependencies
        if i > 0 and random.random() < params['dep_prob']:
            num_deps = min(i, random.randint(1, params['max_deps']))
            dependencies = random.sample(task_ids[:i], num_deps)
            task_data['dependencies'] = dependencies
        else:
            task_data['dependencies'] = []
        
        # Add conflicts
        if random.random() < params['conflict_prob']:
            potential_conflicts = [tid for tid in task_ids[:i] if tid != task_id]
            if potential_conflicts:
                num_conflicts = min(len(potential_conflicts), random.randint(1, 3))
                conflicts = random.sample(potential_conflicts, num_conflicts)
                task_data['conflicts'] = conflicts
            else:
                task_data['conflicts'] = []
        else:
            task_data['conflicts'] = []
        
        # Add resource requirements
        resource_types = ['cpu', 'memory', 'storage', 'network']
        num_resource_reqs = random.randint(1, 3)
        required_resources = {}
        
        for _ in range(num_resource_reqs):
            resource_type = random.choice(resource_types)
            amount = random.randint(1, 4)
            required_resources[resource_type] = amount
        
        task_data['required_resources'] = required_resources
        task_data['preferred_resources'] = {}
        
        # Quantum properties
        task_data['superposition_states'] = []
        task_data['entanglement_partners'] = []
        task_data['quantum_weight'] = random.uniform(0.1, 1.0)
        
        # Metadata
        task_data['tags'] = [complexity, f'batch-{i//50}']
        task_data['attributes'] = {}
        
        tasks_data.append(task_data)
    
    # Generate resources
    resources_data = []
    resource_types = ['cpu', 'memory', 'storage', 'network', 'compute-cluster', 'gpu']
    
    for i in range(resources):
        resource_type = resource_types[i % len(resource_types)]
        
        resource_data = {
            'id': f'resource-{resource_type}-{i+1:02d}',
            'name': f'{resource_type.title()} Resource {i+1}',
            'capacity': random.randint(2, 10),
            'available_capacity': None,  # Will default to capacity
            'cost_per_minute': random.uniform(0.1, 2.0),
            'attributes': {
                'type': resource_type,
                'location': random.choice(['us-east', 'us-west', 'eu-central', 'asia-pacific'])
            }
        }
        
        resources_data.append(resource_data)
    
    # Save to files
    Path(output_tasks).parent.mkdir(parents=True, exist_ok=True)
    Path(output_resources).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_tasks, 'w') as f:
        json.dump(tasks_data, f, indent=2, default=str)
    
    with open(output_resources, 'w') as f:
        json.dump(resources_data, f, indent=2)
    
    click.echo(f"Generated {num_tasks} tasks -> {output_tasks}")
    click.echo(f"Generated {resources} resources -> {output_resources}")


@main.command()
@click.argument('schedule_file', type=click.Path(exists=True))
@click.option('--simulate', is_flag=True, help='Simulate execution without running tasks')
@click.option('--monitor', is_flag=True, help='Monitor execution progress')
@click.pass_context
def execute(ctx, schedule_file, simulate, monitor):
    """Execute a previously generated schedule.
    
    Takes a schedule file and executes or simulates the task execution
    according to the optimized schedule.
    """
    with LogContext(operation="execute"):
        try:
            # Load schedule
            with open(schedule_file, 'r') as f:
                schedule_data = json.load(f)
            
            if 'schedule' not in schedule_data:
                click.echo("Invalid schedule file format", err=True)
                sys.exit(1)
            
            # Create scheduler result object
            result = SchedulingResult(
                schedule=schedule_data['schedule'],
                total_completion_time=schedule_data['total_completion_time'],
                resource_utilization=schedule_data['resource_utilization'],
                optimization_metrics=schedule_data.get('optimization_metrics', {}),
                quantum_metrics=schedule_data.get('quantum_metrics', {}),
                conflicts_resolved=schedule_data.get('conflicts_resolved', 0),
                dependencies_satisfied=schedule_data.get('dependencies_satisfied', 0),
                success=schedule_data.get('success', True)
            )
            
            # Initialize scheduler (minimal setup for execution)
            scheduler = QuantumInspiredScheduler()
            
            # Execute schedule
            logger.info(f"{'Simulating' if simulate else 'Executing'} schedule with {len(result.schedule)} tasks")
            
            execution_result = scheduler.execute_schedule(result, simulate=simulate)
            
            if not execution_result.get('success', False):
                click.echo(f"Execution failed: {execution_result.get('error', 'Unknown error')}", err=True)
                sys.exit(1)
            
            # Display results
            click.echo("\n" + "="*50)
            click.echo(f"SCHEDULE {'SIMULATION' if simulate else 'EXECUTION'} RESULTS")
            click.echo("="*50)
            
            if simulate:
                click.echo(f"Simulated {len(execution_result.get('execution_log', []))} tasks")
                click.echo(f"Total simulated time: {execution_result.get('total_simulated_time', 0)} minutes")
            else:
                click.echo(f"Started {execution_result.get('tasks_started', 0)} tasks")
            
            # Show execution log
            execution_log = execution_result.get('execution_log', [])
            if execution_log:
                click.echo("\nExecution Log:")
                for entry in execution_log[-10:]:  # Show last 10 entries
                    if simulate:
                        click.echo(f"  {entry['task_name']}: {entry['simulated_start']} -> {entry['simulated_end']}")
                    else:
                        click.echo(f"  {entry['task_id']}: {entry['action']} at {entry['timestamp']}")
            
        except Exception as e:
            logger.error(f"Schedule execution failed: {e}")
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)


@main.command()
@click.option('--errors', is_flag=True, help='Show error statistics')
@click.option('--performance', is_flag=True, help='Show performance metrics')
@click.option('--scaling', is_flag=True, help='Show scaling metrics')
@click.option('--export', type=click.Path(), help='Export metrics to file')
@click.pass_context
def status(ctx, errors, performance, scaling, export):
    """Show system status and metrics.
    
    Displays comprehensive status information about the quantum planner
    including error statistics, performance metrics, and scaling status.
    """
    status_data = {
        'timestamp': datetime.now().isoformat(),
        'quantum_planner_version': '1.0.0'
    }
    
    # Error statistics
    if errors:
        error_handler = get_error_handler()
        error_stats = error_handler.get_error_statistics()
        status_data['error_statistics'] = error_stats
        
        click.echo("Error Statistics:")
        click.echo(f"  Total Errors: {error_stats.get('total_errors', 0)}")
        click.echo(f"  Recent Errors (1h): {error_stats.get('recent_errors_1h', 0)}")
        click.echo(f"  Recovery Rate: {error_stats.get('recovery_rate_percent', 0):.1f}%")
        click.echo()
    
    # Performance metrics
    if performance and ctx.obj.get('performance_monitor'):
        perf_monitor = ctx.obj['performance_monitor']
        perf_metrics = perf_monitor.get_metrics_summary()
        status_data['performance_metrics'] = perf_metrics
        
        click.echo("Performance Metrics:")
        system_metrics = perf_metrics.get('system', {})
        click.echo(f"  Memory Usage: {system_metrics.get('memory_percent', 0):.1f}%")
        click.echo(f"  CPU Usage: {system_metrics.get('cpu_percent', 0):.1f}%")
        
        operations = perf_metrics.get('operations', {})
        if operations:
            click.echo("  Operations:")
            for op_name, op_stats in operations.items():
                click.echo(f"    {op_name}: {op_stats['count']} calls, {op_stats['avg_time']:.3f}s avg")
        click.echo()
    
    # Scaling metrics
    if scaling and ctx.obj.get('scaling_orchestrator'):
        scaling_orchestrator = ctx.obj['scaling_orchestrator']
        scaling_metrics = scaling_orchestrator.optimize_system_resources()
        status_data['scaling_metrics'] = scaling_metrics
        
        click.echo("Scaling Metrics:")
        worker_metrics = scaling_metrics.get('worker_metrics', {})
        click.echo(f"  Active Workers: {worker_metrics.get('active_workers', 0)}")
        click.echo(f"  Max Workers: {worker_metrics.get('max_workers', 0)}")
        click.echo(f"  Utilization: {worker_metrics.get('utilization_percent', 0):.1f}%")
        
        recommendations = scaling_orchestrator.get_scaling_recommendations()
        if recommendations:
            click.echo("  Recommendations:")
            for rec in recommendations:
                click.echo(f"    • {rec}")
        click.echo()
    
    # Export to file
    if export:
        Path(export).parent.mkdir(parents=True, exist_ok=True)
        with open(export, 'w') as f:
            json.dump(status_data, f, indent=2, default=str)
        click.echo(f"Status exported to {export}")


@main.command()
@click.option('--level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Log level to show')
@click.option('--count', default=50, help='Number of recent logs to show')
@click.option('--export', type=click.Path(), help='Export logs to file')
@click.pass_context
def logs(ctx, level, count, export):
    """Show recent logs and system activity.
    
    Displays recent log entries from the quantum planner with filtering
    and export capabilities.
    """
    from .utils.logging_config import get_logging_config
    
    logging_config = get_logging_config()
    if not logging_config:
        click.echo("Logging not configured", err=True)
        return
    
    recent_logs = logging_config.get_recent_logs(count=count, level=level)
    
    click.echo(f"Recent Logs ({level} and above):")
    click.echo("="*60)
    
    for log_entry in recent_logs:
        timestamp = log_entry['timestamp'][:19]  # Remove microseconds
        level_str = log_entry['level']
        logger_name = log_entry['logger'].split('.')[-1]  # Short name
        message = log_entry['message']
        
        click.echo(f"{timestamp} | {level_str:8} | {logger_name:15} | {message}")
    
    if export:
        logging_config.export_logs(export, format_type="json", level=level, count=count)
        click.echo(f"\nLogs exported to {export}")


@main.command()
@click.pass_context
def clean(ctx):
    """Clean up temporary files and optimize system resources.
    
    Performs cleanup operations to free up system resources and
    optimize performance.
    """
    click.echo("Cleaning up system resources...")
    
    # Memory optimization
    if ctx.obj.get('performance_monitor'):
        perf_monitor = ctx.obj['performance_monitor']
        optimization_result = perf_monitor.optimize_memory()
        click.echo(f"Memory optimization: {optimization_result}")
    
    # Clear error history
    error_handler = get_error_handler()
    error_handler.clear_error_history()
    click.echo("Cleared error history")
    
    # Clean log files older than 7 days
    logs_dir = Path("logs/quantum_planner")
    if logs_dir.exists():
        cutoff_date = datetime.now() - timedelta(days=7)
        cleaned_files = 0
        
        for log_file in logs_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                cleaned_files += 1
        
        click.echo(f"Cleaned {cleaned_files} old log files")
    
    click.echo("Cleanup completed")


class nullcontext:
    """Null context manager for compatibility."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


if __name__ == '__main__':
    main()