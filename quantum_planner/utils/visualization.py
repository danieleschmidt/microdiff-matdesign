"""Visualization utilities for quantum task scheduling."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import seaborn as sns
import networkx as nx
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Set matplotlib backend for server environments
plt.switch_backend('Agg')

# Configure seaborn style
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')


class ScheduleVisualizer:
    """Create visualizations for quantum task scheduling results."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8),
                 color_palette: str = "husl"):
        """Initialize visualizer."""
        self.figsize = figsize
        self.color_palette = color_palette
        self.colors = sns.color_palette(color_palette, 10)
        
        # Task status colors
        self.status_colors = {
            'pending': '#FFA500',      # Orange
            'in_progress': '#4169E1',  # Royal Blue
            'completed': '#32CD32',    # Lime Green
            'blocked': '#DC143C',      # Crimson
            'cancelled': '#708090',    # Slate Gray
            'failed': '#8B0000'        # Dark Red
        }
        
        # Priority colors
        self.priority_colors = {
            'CRITICAL': '#8B0000',     # Dark Red
            'HIGH': '#FF4500',         # Orange Red
            'MEDIUM': '#FFD700',       # Gold
            'LOW': '#90EE90',          # Light Green
            'DEFERRED': '#D3D3D3'      # Light Gray
        }
        
        logger.info("Initialized ScheduleVisualizer")
    
    def create_gantt_chart(self, schedule: Dict[str, Dict[str, Any]], 
                          output_path: str, title: str = "Task Schedule") -> None:
        """Create Gantt chart visualization of the schedule."""
        if not schedule:
            logger.warning("Empty schedule provided for Gantt chart")
            return
        
        try:
            fig, ax = plt.subplots(figsize=(15, max(8, len(schedule) * 0.5)))
            
            # Prepare data
            tasks = []
            resources = set()
            
            for task_id, task_info in schedule.items():
                tasks.append({
                    'task_id': task_id,
                    'task_name': task_info.get('task_name', task_id[:10]),
                    'start_time': task_info['start_time'],
                    'duration': task_info['duration'],
                    'resource_id': task_info.get('resource_id', 'default'),
                    'priority': task_info.get('priority', 'MEDIUM')
                })
                resources.add(task_info.get('resource_id', 'default'))
            
            # Sort tasks by start time
            tasks.sort(key=lambda x: x['start_time'])
            
            # Create resource color mapping
            resource_colors = {}
            for i, resource in enumerate(sorted(resources)):
                resource_colors[resource] = self.colors[i % len(self.colors)]
            
            # Plot bars
            y_positions = range(len(tasks))
            
            for i, task in enumerate(tasks):
                # Bar color based on priority
                color = self.priority_colors.get(task['priority'], '#87CEEB')
                
                # Create bar
                bar = ax.barh(i, task['duration'], left=task['start_time'],
                            height=0.6, color=color, alpha=0.7,
                            edgecolor='black', linewidth=0.5)
                
                # Add task label
                ax.text(task['start_time'] + task['duration']/2, i,
                       task['task_name'], ha='center', va='center',
                       fontsize=8, weight='bold')
                
                # Add resource indicator
                ax.text(task['start_time'] - 5, i, task['resource_id'],
                       ha='right', va='center', fontsize=7,
                       color=resource_colors[task['resource_id']])
            
            # Formatting
            ax.set_yticks(y_positions)
            ax.set_yticklabels([task['task_name'] for task in tasks])
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('Tasks')
            ax.set_title(title, fontsize=16, weight='bold')
            
            # Add legend for priorities
            priority_patches = [mpatches.Patch(color=color, label=priority)
                              for priority, color in self.priority_colors.items()
                              if any(t['priority'] == priority for t in tasks)]
            
            if priority_patches:
                ax.legend(handles=priority_patches, loc='upper right',
                         title='Priority', bbox_to_anchor=(1.15, 1))
            
            # Grid and layout
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gantt chart saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating Gantt chart: {e}")
            raise
    
    def create_resource_utilization_chart(self, resource_utilization: Dict[str, float],
                                        output_path: str, 
                                        title: str = "Resource Utilization") -> None:
        """Create resource utilization visualization."""
        if not resource_utilization:
            logger.warning("Empty resource utilization data")
            return
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            resources = list(resource_utilization.keys())
            utilizations = [resource_utilization[r] * 100 for r in resources]  # Convert to percentage
            
            # Bar chart
            bars = ax1.bar(resources, utilizations, color=self.colors[:len(resources)], alpha=0.7)
            ax1.set_ylabel('Utilization (%)')
            ax1.set_title('Resource Utilization')
            ax1.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar, util in zip(bars, utilizations):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{util:.1f}%', ha='center', va='bottom')
            
            # Color bars based on utilization level
            for bar, util in zip(bars, utilizations):
                if util > 90:
                    bar.set_color('#FF4444')  # Over-utilized - red
                elif util > 70:
                    bar.set_color('#FFA500')  # High utilization - orange
                elif util < 30:
                    bar.set_color('#90EE90')  # Under-utilized - light green
                else:
                    bar.set_color('#4169E1')  # Good utilization - blue
            
            # Pie chart
            ax2.pie(utilizations, labels=resources, autopct='%1.1f%%',
                   colors=self.colors[:len(resources)], startangle=90)
            ax2.set_title('Resource Utilization Distribution')
            
            fig.suptitle(title, fontsize=16, weight='bold')
            plt.tight_layout()
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Resource utilization chart saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating resource utilization chart: {e}")
            raise
    
    def create_dependency_graph(self, dependency_graph: Dict[str, Any],
                              output_path: str,
                              title: str = "Task Dependencies") -> None:
        """Create task dependency graph visualization."""
        if not dependency_graph.get('nodes'):
            logger.warning("Empty dependency graph data")
            return
        
        try:
            # Create NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes
            node_colors = []
            node_sizes = []
            labels = {}
            
            for node in dependency_graph['nodes']:
                node_id = node['id']
                G.add_node(node_id, **node)
                
                # Color by priority
                priority = node.get('priority', 'MEDIUM')
                node_colors.append(self.priority_colors.get(priority, '#87CEEB'))
                
                # Size by quantum weight
                quantum_weight = node.get('quantum_weight', 1.0)
                node_sizes.append(300 + quantum_weight * 500)
                
                # Label
                labels[node_id] = node.get('name', node_id[:8])
            
            # Add edges
            for edge in dependency_graph.get('edges', []):
                source = edge['source']
                target = edge['target']
                edge_type = edge.get('type', 'dependency')
                
                if source in G.nodes and target in G.nodes:
                    G.add_edge(source, target, type=edge_type)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Layout
            try:
                pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
            except:
                pos = nx.random_layout(G, seed=42)
            
            # Draw dependency edges
            dependency_edges = [(u, v) for u, v, d in G.edges(data=True) 
                              if d.get('type') == 'dependency']
            if dependency_edges:
                nx.draw_networkx_edges(G, pos, edgelist=dependency_edges,
                                     edge_color='blue', arrows=True,
                                     arrowsize=20, alpha=0.6, width=2)
            
            # Draw conflict edges
            conflict_edges = [(u, v) for u, v, d in G.edges(data=True)
                            if d.get('type') == 'conflict']
            if conflict_edges:
                nx.draw_networkx_edges(G, pos, edgelist=conflict_edges,
                                     edge_color='red', arrows=False,
                                     style='dashed', alpha=0.5, width=1)
            
            # Draw entanglement edges
            entanglement_edges = [(u, v) for u, v, d in G.edges(data=True)
                                if d.get('type') == 'entanglement']
            if entanglement_edges:
                nx.draw_networkx_edges(G, pos, edgelist=entanglement_edges,
                                     edge_color='green', arrows=False,
                                     style='dotted', alpha=0.7, width=1.5)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                                 node_size=node_sizes, alpha=0.8)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
            
            # Legend
            legend_elements = [
                mpatches.Patch(color='blue', label='Dependencies'),
                mpatches.Patch(color='red', label='Conflicts'),
                mpatches.Patch(color='green', label='Entanglements')
            ]
            
            # Add priority legend
            for priority, color in self.priority_colors.items():
                if any(node.get('priority') == priority for node in dependency_graph['nodes']):
                    legend_elements.append(mpatches.Patch(color=color, label=f'{priority} Priority'))
            
            ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            
            ax.set_title(title, fontsize=16, weight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Dependency graph saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating dependency graph: {e}")
            raise
    
    def create_performance_metrics_dashboard(self, metrics: Dict[str, Any],
                                           output_path: str,
                                           title: str = "Performance Dashboard") -> None:
        """Create comprehensive performance metrics dashboard."""
        try:
            fig = plt.figure(figsize=(20, 12))
            
            # Create grid layout
            gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
            
            # 1. Optimization metrics over time
            if 'optimization_history' in metrics:
                ax1 = fig.add_subplot(gs[0, :2])
                history = metrics['optimization_history']
                iterations = range(len(history))
                ax1.plot(iterations, history, 'b-', linewidth=2)
                ax1.set_title('Optimization Progress')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Energy/Cost')
                ax1.grid(True, alpha=0.3)
            
            # 2. System resource usage
            if 'system_metrics' in metrics:
                ax2 = fig.add_subplot(gs[0, 2:])
                sys_metrics = metrics['system_metrics']
                
                categories = ['Memory', 'CPU', 'Disk']
                values = [
                    sys_metrics.get('memory_percent', 0),
                    sys_metrics.get('cpu_percent', 0),
                    sys_metrics.get('disk_usage_percent', 0)
                ]
                
                bars = ax2.bar(categories, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                ax2.set_title('System Resource Usage')
                ax2.set_ylabel('Usage (%)')
                ax2.set_ylim(0, 100)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{value:.1f}%', ha='center', va='bottom')
            
            # 3. Quantum metrics
            if 'quantum_metrics' in metrics:
                ax3 = fig.add_subplot(gs[1, :2])
                quantum = metrics['quantum_metrics']
                
                # Plot quantum energy distribution
                if 'energy_history' in quantum:
                    energies = quantum['energy_history']
                    ax3.hist(energies, bins=20, alpha=0.7, color='purple', edgecolor='black')
                    ax3.set_title('Quantum Energy Distribution')
                    ax3.set_xlabel('Energy Level')
                    ax3.set_ylabel('Frequency')
                    ax3.axvline(np.mean(energies), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(energies):.3f}')
                    ax3.legend()
            
            # 4. Task completion statistics
            if 'task_stats' in metrics:
                ax4 = fig.add_subplot(gs[1, 2:])
                task_stats = metrics['task_stats']
                
                # Pie chart of task statuses
                statuses = list(task_stats.keys())
                counts = list(task_stats.values())
                colors = [self.status_colors.get(status, '#87CEEB') for status in statuses]
                
                ax4.pie(counts, labels=statuses, colors=colors, autopct='%1.1f%%')
                ax4.set_title('Task Status Distribution')
            
            # 5. Performance trends
            if 'performance_trends' in metrics:
                ax5 = fig.add_subplot(gs[2, :])
                trends = metrics['performance_trends']
                
                timestamps = [datetime.fromisoformat(t) for t in trends['timestamps']]
                
                # Multiple metrics on same plot
                ax5_twin = ax5.twinx()
                
                line1 = ax5.plot(timestamps, trends.get('response_times', []), 
                               'b-', label='Response Time', linewidth=2)
                line2 = ax5_twin.plot(timestamps, trends.get('throughput', []), 
                                    'r-', label='Throughput', linewidth=2)
                
                ax5.set_xlabel('Time')
                ax5.set_ylabel('Response Time (ms)', color='b')
                ax5_twin.set_ylabel('Throughput (tasks/sec)', color='r')
                
                # Format x-axis
                ax5.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax5.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax5.legend(lines, labels, loc='upper left')
                
                ax5.set_title('Performance Trends')
                ax5.grid(True, alpha=0.3)
            
            fig.suptitle(title, fontsize=20, weight='bold')
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance dashboard saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating performance dashboard: {e}")
            raise
    
    def create_quantum_state_visualization(self, quantum_states: List[Dict[str, Any]],
                                         output_path: str,
                                         title: str = "Quantum States") -> None:
        """Create quantum state visualization."""
        if not quantum_states:
            logger.warning("Empty quantum states data")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. State probability distribution
            probabilities = [state.get('probability', 0) for state in quantum_states]
            energies = [state.get('energy', 0) for state in quantum_states]
            
            ax1.scatter(range(len(probabilities)), probabilities, 
                       c=energies, cmap='viridis', s=50, alpha=0.7)
            ax1.set_title('Quantum State Probabilities')
            ax1.set_xlabel('State Index')
            ax1.set_ylabel('Probability')
            ax1.grid(True, alpha=0.3)
            
            # 2. Energy distribution
            ax2.hist(energies, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax2.set_title('Energy Distribution')
            ax2.set_xlabel('Energy')
            ax2.set_ylabel('Frequency')
            ax2.axvline(np.mean(energies), color='red', linestyle='--',
                       label=f'Mean: {np.mean(energies):.3f}')
            ax2.legend()
            
            # 3. Probability vs Energy scatter
            ax3.scatter(probabilities, energies, alpha=0.6, s=50)
            ax3.set_title('Probability vs Energy')
            ax3.set_xlabel('Probability')
            ax3.set_ylabel('Energy')
            ax3.grid(True, alpha=0.3)
            
            # Add trend line
            if len(probabilities) > 1:
                z = np.polyfit(probabilities, energies, 1)
                p = np.poly1d(z)
                ax3.plot(probabilities, p(probabilities), "r--", alpha=0.8)
            
            # 4. Quantum metrics over time
            if len(quantum_states) > 10:
                # Group into time windows
                window_size = len(quantum_states) // 10
                windowed_probs = []
                windowed_energies = []
                
                for i in range(0, len(quantum_states), window_size):
                    window = quantum_states[i:i+window_size]
                    avg_prob = np.mean([s.get('probability', 0) for s in window])
                    avg_energy = np.mean([s.get('energy', 0) for s in window])
                    windowed_probs.append(avg_prob)
                    windowed_energies.append(avg_energy)
                
                ax4_twin = ax4.twinx()
                
                line1 = ax4.plot(windowed_probs, 'b-', label='Avg Probability', linewidth=2)
                line2 = ax4_twin.plot(windowed_energies, 'r-', label='Avg Energy', linewidth=2)
                
                ax4.set_ylabel('Probability', color='b')
                ax4_twin.set_ylabel('Energy', color='r')
                ax4.set_xlabel('Time Window')
                ax4.set_title('Quantum Evolution')
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax4.legend(lines, labels, loc='upper right')
            
            fig.suptitle(title, fontsize=16, weight='bold')
            plt.tight_layout()
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Quantum state visualization saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating quantum state visualization: {e}")
            raise
    
    def create_scaling_analysis_chart(self, scaling_data: Dict[str, List[float]],
                                    output_path: str,
                                    title: str = "Scaling Analysis") -> None:
        """Create scaling performance analysis chart."""
        if not scaling_data:
            logger.warning("Empty scaling data")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            
            # 1. Processing time vs task count
            if 'task_counts' in scaling_data and 'processing_times' in scaling_data:
                ax1.plot(scaling_data['task_counts'], scaling_data['processing_times'], 
                        'bo-', linewidth=2, markersize=6)
                ax1.set_title('Processing Time vs Task Count')
                ax1.set_xlabel('Number of Tasks')
                ax1.set_ylabel('Processing Time (s)')
                ax1.grid(True, alpha=0.3)
                
                # Add ideal linear scaling reference
                if scaling_data['task_counts']:
                    max_tasks = max(scaling_data['task_counts'])
                    max_time = max(scaling_data['processing_times'])
                    ideal_times = [(t / max_tasks) * max_time for t in scaling_data['task_counts']]
                    ax1.plot(scaling_data['task_counts'], ideal_times, 
                            'r--', alpha=0.5, label='Ideal Linear')
                    ax1.legend()
            
            # 2. Worker utilization over time
            if 'worker_utilization_history' in scaling_data:
                utilization_history = scaling_data['worker_utilization_history']
                ax2.plot(range(len(utilization_history)), utilization_history, 
                        'g-', linewidth=2)
                ax2.set_title('Worker Utilization Over Time')
                ax2.set_xlabel('Time Steps')
                ax2.set_ylabel('Utilization (%)')
                ax2.set_ylim(0, 100)
                ax2.grid(True, alpha=0.3)
                
                # Add target utilization line
                ax2.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Target (80%)')
                ax2.legend()
            
            # 3. Memory usage scaling
            if 'memory_usage' in scaling_data:
                memory_data = scaling_data['memory_usage']
                ax3.bar(range(len(memory_data)), memory_data, color='orange', alpha=0.7)
                ax3.set_title('Memory Usage by Worker')
                ax3.set_xlabel('Worker ID')
                ax3.set_ylabel('Memory Usage (MB)')
                ax3.grid(True, alpha=0.3)
            
            # 4. Throughput analysis
            if 'throughput_data' in scaling_data:
                throughput = scaling_data['throughput_data']
                ax4.plot(throughput['timestamps'], throughput['values'], 
                        'purple', linewidth=2)
                ax4.set_title('System Throughput')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Tasks/Second')
                ax4.grid(True, alpha=0.3)
                
                # Add average line
                avg_throughput = np.mean(throughput['values'])
                ax4.axhline(y=avg_throughput, color='red', linestyle='--', 
                           alpha=0.7, label=f'Average: {avg_throughput:.2f}')
                ax4.legend()
            
            fig.suptitle(title, fontsize=16, weight='bold')
            plt.tight_layout()
            
            # Save
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Scaling analysis chart saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating scaling analysis chart: {e}")
            raise