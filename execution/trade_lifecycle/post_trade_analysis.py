"""
Post-trade analysis module for assessing trading performance.
Analyzes execution quality, costs, and performance after trade completion.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PostTradeAnalyzer:
    """Analyzer for post-trade performance assessment."""
    
    def __init__(self, 
                 benchmark_window_minutes: int = 15,
                 twap_comparison: bool = True,
                 vwap_comparison: bool = True):
        """
        Initialize the post-trade analyzer.
        
        Args:
            benchmark_window_minutes: Time window for benchmark calculations
            twap_comparison: Whether to compare with TWAP benchmark
            vwap_comparison: Whether to compare with VWAP benchmark
        """
        self.benchmark_window_minutes = benchmark_window_minutes
        self.twap_comparison = twap_comparison
        self.vwap_comparison = vwap_comparison
    
    def analyze_execution(self, 
                         execution_data: Dict[str, Any],
                         market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a completed trade execution.
        
        Args:
            execution_data: Completed execution data
            market_data: Market data dictionary with historical data
            
        Returns:
            Analysis results with performance metrics
        """
        logger.info(f"Performing post-trade analysis for execution {execution_data.get('execution_id', 'unknown')}")
        
        # Initialize results dictionary
        results = {
            'execution_id': execution_data.get('execution_id', 'unknown'),
            'symbol': execution_data.get('symbol', 'unknown'),
            'analysis_time': datetime.now().isoformat(),
            'metrics': {},
            'benchmarks': {},
            'costs': {},
            'commentary': []
        }
        
        # Extract basic execution information
        try:
            side = execution_data.get('order_details', {}).get('side', 'buy').lower()
            filled_quantity = execution_data.get('filled_quantity', 0)
            avg_price = execution_data.get('avg_fill_price', 0)
            expected_price = execution_data.get('expected_price', 0)
            fills = execution_data.get('fills', [])
            start_time = execution_data.get('start_time')
            completion_time = execution_data.get('completion_time') or execution_data.get('last_update_time')
            
            # Convert string times to datetime if necessary
            if isinstance(start_time, str):
                start_time = datetime.fromisoformat(start_time)
            if isinstance(completion_time, str):
                completion_time = datetime.fromisoformat(completion_time)
            
            # Calculate execution duration
            if start_time and completion_time:
                duration_seconds = (completion_time - start_time).total_seconds()
            else:
                duration_seconds = 0
            
            # Basic execution metrics
            results['metrics']['filled_quantity'] = filled_quantity
            results['metrics']['avg_price'] = avg_price
            results['metrics']['expected_price'] = expected_price
            results['metrics']['duration_seconds'] = duration_seconds
            results['metrics']['fill_count'] = len(fills)
            
            # Calculate price metrics
            if expected_price > 0 and avg_price > 0:
                if side == 'buy':
                    price_improvement = expected_price - avg_price
                else:
                    price_improvement = avg_price - expected_price
                
                price_improvement_pct = price_improvement / expected_price * 100
                results['metrics']['price_improvement'] = price_improvement
                results['metrics']['price_improvement_pct'] = price_improvement_pct
            
            # Calculate transaction costs
            results['costs'] = self._calculate_transaction_costs(execution_data, market_data)
            
            # Calculate benchmark comparisons
            results['benchmarks'] = self._calculate_benchmarks(
                execution_data, market_data, start_time, completion_time)
            
            # Generate performance commentary
            results['commentary'] = self._generate_commentary(results)
            
            # Calculate implementation shortfall
            results['metrics']['implementation_shortfall'] = self._calculate_implementation_shortfall(
                execution_data, market_data)
            
            # Calculate additional metrics
            results['metrics']['fill_uniformity'] = self._calculate_fill_uniformity(fills)
            results['metrics']['price_variance'] = self._calculate_price_variance(fills)
            
            logger.info(f"Post-trade analysis completed for {results['execution_id']}")
            
        except Exception as e:
            logger.error(f"Error in post-trade analysis: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def compare_executions(self, 
                          executions: List[Dict[str, Any]],
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare multiple executions against each other.
        
        Args:
            executions: List of execution data dictionaries
            market_data: Market data dictionary
            
        Returns:
            Comparison analysis results
        """
        results = {
            'analysis_time': datetime.now().isoformat(),
            'execution_count': len(executions),
            'individual_analyses': [],
            'comparison': {},
            'summary': {}
        }
        
        # Analyze each execution
        for execution in executions:
            analysis = self.analyze_execution(execution, market_data)
            results['individual_analyses'].append(analysis)
        
        # Compare executions if there are more than one
        if len(executions) > 1:
            # Extract metrics for comparison
            price_improvements = []
            impl_shortfalls = []
            vwap_performances = []
            
            for analysis in results['individual_analyses']:
                price_improvements.append(analysis['metrics'].get('price_improvement_pct', 0))
                impl_shortfalls.append(analysis['metrics'].get('implementation_shortfall', 0))
                vwap_performances.append(analysis['benchmarks'].get('vwap_performance', 0))
            
            # Calculate comparison statistics
            results['comparison']['price_improvement'] = {
                'mean': np.mean(price_improvements),
                'std_dev': np.std(price_improvements),
                'min': np.min(price_improvements),
                'max': np.max(price_improvements)
            }
            
            results['comparison']['implementation_shortfall'] = {
                'mean': np.mean(impl_shortfalls),
                'std_dev': np.std(impl_shortfalls),
                'min': np.min(impl_shortfalls),
                'max': np.max(impl_shortfalls)
            }
            
            results['comparison']['vwap_performance'] = {
                'mean': np.mean(vwap_performances),
                'std_dev': np.std(vwap_performances),
                'min': np.min(vwap_performances),
                'max': np.max(vwap_performances)
            }
            
            # Identify best and worst executions
            best_idx = np.argmax(price_improvements)
            worst_idx = np.argmin(price_improvements)
            
            results['summary']['best_execution'] = {
                'execution_id': executions[best_idx].get('execution_id', 'unknown'),
                'price_improvement_pct': price_improvements[best_idx]
            }
            
            results['summary']['worst_execution'] = {
                'execution_id': executions[worst_idx].get('execution_id', 'unknown'),
                'price_improvement_pct': price_improvements[worst_idx]
            }
        
        return results
    
    def analyze_algo_performance(self, 
                               executions: List[Dict[str, Any]],
                               algo_name: str,
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the performance of a specific execution algorithm.
        
        Args:
            executions: List of executions using this algorithm
            algo_name: Algorithm name
            market_data: Market data dictionary
            
        Returns:
            Algorithm performance analysis
        """
        # Filter executions by algorithm
        algo_executions = [e for e in executions if e.get('algorithm', '').lower() == algo_name.lower()]
        
        results = {
            'algorithm': algo_name,
            'analysis_time': datetime.now().isoformat(),
            'execution_count': len(algo_executions),
            'performance_metrics': {},
            'market_condition_analysis': {},
            'recommendations': []
        }
        
        if not algo_executions:
            results['recommendations'].append("No executions found for this algorithm.")
            return results
        
        # Analyze individual executions
        analyses = [self.analyze_execution(exec_data, market_data) for exec_data in algo_executions]
        
        # Extract performance metrics
        price_improvements = [a['metrics'].get('price_improvement_pct', 0) for a in analyses]
        vwap_performances = [a['benchmarks'].get('vwap_performance', 0) for a in analyses]
        durations = [a['metrics'].get('duration_seconds', 0) for a in analyses]
        costs = [sum(a['costs'].values()) for a in analyses]
        
        # Calculate aggregate metrics
        results['performance_metrics']['avg_price_improvement'] = np.mean(price_improvements)
        results['performance_metrics']['avg_vwap_performance'] = np.mean(vwap_performances)
        results['performance_metrics']['avg_duration'] = np.mean(durations)
        results['performance_metrics']['avg_cost'] = np.mean(costs)
        results['performance_metrics']['success_rate'] = np.mean([1 if p > 0 else 0 for p in price_improvements])
        
        # Analyze performance in different market conditions
        # (Simplified - would use more sophisticated market regime detection in real implementation)
        high_vol_indices = []
        low_vol_indices = []
        
        for i, exec_data in enumerate(algo_executions):
            if 'market_conditions' in exec_data and 'volatility' in exec_data['market_conditions']:
                if exec_data['market_conditions']['volatility'] > 0.3:  # Arbitrary threshold
                    high_vol_indices.append(i)
                else:
                    low_vol_indices.append(i)
        
        # Compare performance in different market conditions
        if high_vol_indices and low_vol_indices:
            high_vol_improvement = np.mean([price_improvements[i] for i in high_vol_indices])
            low_vol_improvement = np.mean([price_improvements[i] for i in low_vol_indices])
            
            results['market_condition_analysis']['high_volatility'] = {
                'count': len(high_vol_indices),
                'avg_price_improvement': high_vol_improvement
            }
            
            results['market_condition_analysis']['low_volatility'] = {
                'count': len(low_vol_indices),
                'avg_price_improvement': low_vol_improvement
            }
            
            # Generate recommendations
            if high_vol_improvement > low_vol_improvement:
                results['recommendations'].append(
                    f"Algorithm {algo_name} performs better in high volatility environments. "
                    f"Consider preferring this algorithm during volatile periods."
                )
            else:
                results['recommendations'].append(
                    f"Algorithm {algo_name} performs better in low volatility environments. "
                    f"Consider other algorithms during high volatility periods."
                )
        
        # Add general recommendations
        if results['performance_metrics']['avg_price_improvement'] < 0:
            results['recommendations'].append(
                f"Algorithm {algo_name} is underperforming expectations. Consider reviewing parameters."
            )
        
        return results
    
    def _calculate_transaction_costs(self, 
                                    execution_data: Dict[str, Any],
                                    market_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate detailed transaction costs.
        
        Args:
            execution_data: Execution data
            market_data: Market data
            
        Returns:
            Dictionary of transaction costs
        """
        costs = {
            'spread_cost': 0.0,
            'market_impact': 0.0,
            'fee_cost': 0.0,
            'opportunity_cost': 0.0,
            'total_cost': 0.0
        }
        
        # Extract necessary data
        filled_quantity = execution_data.get('filled_quantity', 0)
        avg_price = execution_data.get('avg_fill_price', 0)
        side = execution_data.get('order_details', {}).get('side', 'buy').lower()
        
        # Calculate spread cost
        if 'market_conditions' in execution_data and 'best_bid' in execution_data['market_conditions']:
            best_bid = execution_data['market_conditions'].get('best_bid', 0)
            best_ask = execution_data['market_conditions'].get('best_ask', 0)
            
            if best_bid > 0 and best_ask > 0:
                mid_price = (best_bid + best_ask) / 2
                
                if side == 'buy':
                    spread_cost = (avg_price - mid_price) * filled_quantity
                else:
                    spread_cost = (mid_price - avg_price) * filled_quantity
                
                costs['spread_cost'] = max(0, spread_cost)
        
        # Calculate market impact (simplified)
        if 'expected_price' in execution_data and execution_data['expected_price'] > 0:
            expected_price = execution_data['expected_price']
            if side == 'buy':
                impact = max(0, avg_price - expected_price) * filled_quantity
            else:
                impact = max(0, expected_price - avg_price) * filled_quantity
            
            costs['market_impact'] = impact
        
        # Calculate fee cost (simplified - would use actual fee structure)
        fees_rate = 0.001  # 0.1% fee
        costs['fee_cost'] = avg_price * filled_quantity * fees_rate
        
        # Calculate opportunity cost (simplified)
        # Assumes a theoretical "optimal" execution better than expected by 0.05%
        optimal_improvement = 0.0005
        if 'expected_price' in execution_data and execution_data['expected_price'] > 0:
            if side == 'buy':
                optimal_price = execution_data['expected_price'] * (1 - optimal_improvement)
                opportunity_cost = (avg_price - optimal_price) * filled_quantity
            else:
                optimal_price = execution_data['expected_price'] * (1 + optimal_improvement)
                opportunity_cost = (optimal_price - avg_price) * filled_quantity
            
            costs['opportunity_cost'] = max(0, opportunity_cost)
        
        # Calculate total cost
        costs['total_cost'] = (
            costs['spread_cost'] + 
            costs['market_impact'] + 
            costs['fee_cost'] + 
            costs['opportunity_cost']
        )
        
        return costs
    
    def _calculate_benchmarks(self, 
                             execution_data: Dict[str, Any],
                             market_data: Dict[str, Any],
                             start_time: datetime,
                             completion_time: datetime) -> Dict[str, float]:
        """
        Calculate benchmark comparisons.
        
        Args:
            execution_data: Execution data
            market_data: Market data with price history
            start_time: Execution start time
            completion_time: Execution completion time
            
        Returns:
            Dictionary of benchmark comparisons
        """
        benchmarks = {
            'arrival_performance': 0.0,
            'twap_performance': 0.0,
            'vwap_performance': 0.0
        }
        
        # Extract necessary data
        avg_price = execution_data.get('avg_fill_price', 0)
        side = execution_data.get('order_details', {}).get('side', 'buy').lower()
        
        # Calculate arrival price performance
        if 'expected_price' in execution_data and execution_data['expected_price'] > 0:
            arrival_price = execution_data['expected_price']
            
            if side == 'buy':
                performance = (arrival_price - avg_price) / arrival_price * 100
            else:
                performance = (avg_price - arrival_price) / arrival_price * 100
            
            benchmarks['arrival_performance'] = performance
        
        # Calculate TWAP benchmark
        if 'price_history' in market_data and start_time and completion_time:
            price_history = market_data['price_history']
            
            # Filter price history to execution window
            if isinstance(price_history, list):
                # Simple list of prices case
                twap = np.mean(price_history)
            elif isinstance(price_history, dict) and 'timestamps' in price_history and 'prices' in price_history:
                # Time series data case
                filtered_prices = []
                
                for ts, price in zip(price_history['timestamps'], price_history['prices']):
                    if isinstance(ts, str):
                        ts_dt = datetime.fromisoformat(ts)
                    else:
                        ts_dt = ts
                    
                    if start_time <= ts_dt <= completion_time:
                        filtered_prices.append(price)
                
                twap = np.mean(filtered_prices) if filtered_prices else 0
            else:
                twap = 0
            
            if twap > 0:
                if side == 'buy':
                    performance = (twap - avg_price) / twap * 100
                else:
                    performance = (avg_price - twap) / twap * 100
                
                benchmarks['twap_performance'] = performance
        
        # Calculate VWAP benchmark (simplified)
        if 'vwap' in market_data:
            vwap = market_data['vwap']
            
            if vwap > 0:
                if side == 'buy':
                    performance = (vwap - avg_price) / vwap * 100
                else:
                    performance = (avg_price - vwap) / vwap * 100
                
                benchmarks['vwap_performance'] = performance
        
        return benchmarks
    
    def _calculate_implementation_shortfall(self, 
                                           execution_data: Dict[str, Any],
                                           market_data: Dict[str, Any]) -> float:
        """
        Calculate implementation shortfall.
        
        Args:
            execution_data: Execution data
            market_data: Market data
            
        Returns:
            Implementation shortfall as percentage
        """
        # Extract data
        decision_price = execution_data.get('expected_price', 0)
        avg_price = execution_data.get('avg_fill_price', 0)
        side = execution_data.get('order_details', {}).get('side', 'buy').lower()
        filled_quantity = execution_data.get('filled_quantity', 0)
        
        if decision_price <= 0 or avg_price <= 0 or filled_quantity <= 0:
            return 0.0
        
        # Calculate implementation shortfall
        if side == 'buy':
            shortfall = (avg_price - decision_price) / decision_price * 100
        else:
            shortfall = (decision_price - avg_price) / decision_price * 100
        
        return shortfall
    
    def _calculate_fill_uniformity(self, fills: List[Dict[str, Any]]) -> float:
        """
        Calculate how uniform the fills were over time.
        
        Args:
            fills: List of fills
            
        Returns:
            Uniformity score (0-1, higher is more uniform)
        """
        if not fills or len(fills) < 2:
            return 1.0
        
        # Extract timestamps and quantities
        timestamps = []
        quantities = []
        
        for fill in fills:
            if 'time' in fill and 'quantity' in fill:
                ts = fill['time']
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                
                timestamps.append(ts)
                quantities.append(fill['quantity'])
        
        if not timestamps or len(timestamps) < 2:
            return 1.0
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        # Calculate coefficient of variation for intervals
        cv_intervals = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        # Calculate coefficient of variation for quantities
        cv_quantities = np.std(quantities) / np.mean(quantities) if np.mean(quantities) > 0 else 0
        
        # Combined uniformity score (inverse of average CV, normalized to 0-1)
        avg_cv = (cv_intervals + cv_quantities) / 2
        uniformity_score = 1.0 / (1.0 + avg_cv)
        
        return uniformity_score
    
    def _calculate_price_variance(self, fills: List[Dict[str, Any]]) -> float:
        """
        Calculate the variance in fill prices.
        
        Args:
            fills: List of fills
            
        Returns:
            Price variance
        """
        if not fills:
            return 0.0
        
        prices = [fill.get('price', 0) for fill in fills if 'price' in fill]
        
        if not prices:
            return 0.0
        
        # Calculate variance
        return np.var(prices)
    
    def _generate_commentary(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable commentary based on analysis.
        
        Args:
            results: Analysis results
            
        Returns:
            List of commentary strings
        """
        commentary = []
        
        # Extract metrics
        metrics = results.get('metrics', {})
        benchmarks = results.get('benchmarks', {})
        costs = results.get('costs', {})
        
        # Price improvement commentary
        if 'price_improvement_pct' in metrics:
            improvement = metrics['price_improvement_pct']
            if improvement > 0.5:
                commentary.append(f"Excellent price improvement of {improvement:.2f}% achieved.")
            elif improvement > 0:
                commentary.append(f"Positive price improvement of {improvement:.2f}% achieved.")
            elif improvement > -0.5:
                commentary.append(f"Execution achieved close to expected price with minimal slippage ({improvement:.2f}%).")
            else:
                commentary.append(f"Poor price performance with significant negative slippage of {improvement:.2f}%.")
        
        # Benchmark comparison commentary
        if 'vwap_performance' in benchmarks:
            vwap_perf = benchmarks['vwap_performance']
            if vwap_perf > 0.2:
                commentary.append(f"Execution outperformed VWAP by {vwap_perf:.2f}%.")
            elif vwap_perf > -0.2:
                commentary.append(f"Execution was in line with VWAP (diff: {vwap_perf:.2f}%).")
            else:
                commentary.append(f"Execution underperformed VWAP by {-vwap_perf:.2f}%.")
        
        # Cost commentary
        if 'total_cost' in costs:
            total_cost = costs['total_cost']
            if 'filled_quantity' in metrics and metrics['filled_quantity'] > 0 and 'avg_price' in metrics:
                relative_cost = total_cost / (metrics['filled_quantity'] * metrics['avg_price']) * 100
                
                if relative_cost > 0.5:
                    commentary.append(f"High transaction costs of {relative_cost:.2f}% observed.")
                else:
                    commentary.append(f"Transaction costs were reasonable at {relative_cost:.2f}%.")
        
        # Fill uniformity commentary
        if 'fill_uniformity' in metrics:
            uniformity = metrics['fill_uniformity']
            if uniformity > 0.8:
                commentary.append("Fills were well distributed throughout the execution.")
            elif uniformity > 0.4:
                commentary.append("Fills showed some clustering during execution.")
            else:
                commentary.append("Fills were highly clustered, suggesting potential issues with the execution algorithm.")
        
        # Duration commentary
        if 'duration_seconds' in metrics:
            duration = metrics['duration_seconds']
            if duration < 10:
                commentary.append("Execution completed very quickly.")
            elif duration > 300:  # 5 minutes
                commentary.append(f"Execution took longer than expected ({duration:.1f} seconds).")
        
        return commentary 