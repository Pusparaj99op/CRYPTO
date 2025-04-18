"""
Trade reconciliation module for reconciling trade data between systems.
Ensures consistency between internal records and exchange/broker data.
"""
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradeReconciliation:
    """
    Trade reconciliation system to ensure data consistency between trading systems.
    """
    
    def __init__(self, 
                 tolerance_amount: float = 0.001,
                 tolerance_percentage: float = 0.01,
                 auto_resolve_threshold: float = 0.0001):
        """
        Initialize the trade reconciliation system.
        
        Args:
            tolerance_amount: Absolute tolerance for matching amounts
            tolerance_percentage: Percentage tolerance for matching
            auto_resolve_threshold: Threshold for automatic resolution of small discrepancies
        """
        self.tolerance_amount = tolerance_amount
        self.tolerance_percentage = tolerance_percentage
        self.auto_resolve_threshold = auto_resolve_threshold
        
        # For storing reconciliation results
        self.reconciliation_history: List[Dict[str, Any]] = []
    
    def reconcile_trades(self, 
                        internal_trades: List[Dict[str, Any]],
                        external_trades: List[Dict[str, Any]],
                        match_fields: List[str],
                        amount_field: str = 'quantity',
                        price_field: str = 'price',
                        id_field: str = 'trade_id') -> Dict[str, Any]:
        """
        Reconcile internal trade records with external (exchange/broker) data.
        
        Args:
            internal_trades: List of internal trade records
            external_trades: List of external trade records
            match_fields: List of fields to use for matching records
            amount_field: Field name containing quantity information
            price_field: Field name containing price information
            id_field: Field name containing trade identifier
            
        Returns:
            Reconciliation result dictionary
        """
        logger.info(f"Starting trade reconciliation: {len(internal_trades)} internal trades, "
                  f"{len(external_trades)} external trades")
        
        # Create reconciliation result structure
        result = {
            'reconciliation_id': f"recon_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'matched': [],
            'unmatched_internal': [],
            'unmatched_external': [],
            'amount_mismatches': [],
            'price_mismatches': [],
            'auto_resolved': [],
            'statistics': {},
            'status': 'pending'
        }
        
        # Convert to pandas DataFrames for easier matching
        df_internal = pd.DataFrame(internal_trades)
        df_external = pd.DataFrame(external_trades)
        
        if df_internal.empty or df_external.empty:
            logger.warning("One or both trade sets are empty")
            result['status'] = 'completed'
            result['statistics'] = self._calculate_statistics(result)
            self.reconciliation_history.append(result)
            return result
        
        # Ensure matching fields exist in both DataFrames
        valid_match_fields = [field for field in match_fields 
                             if field in df_internal.columns and field in df_external.columns]
        
        if not valid_match_fields:
            logger.error("No valid matching fields found in both datasets")
            result['status'] = 'failed'
            result['error'] = 'No valid matching fields found'
            self.reconciliation_history.append(result)
            return result
        
        # Create merge keys for matching
        df_internal['_merge_key'] = self._create_merge_keys(df_internal, valid_match_fields)
        df_external['_merge_key'] = self._create_merge_keys(df_external, valid_match_fields)
        
        # Find perfect matches
        internal_matched_indices = []
        external_matched_indices = []
        
        for i, internal_row in df_internal.iterrows():
            internal_key = internal_row['_merge_key']
            
            # Find matching external rows
            matching_indices = df_external[df_external['_merge_key'] == internal_key].index.tolist()
            
            if len(matching_indices) == 1:
                # Single match found
                external_idx = matching_indices[0]
                external_row = df_external.loc[external_idx]
                
                # Check if amounts match within tolerance
                amount_match, amount_diff = self._check_amount_match(
                    internal_row.get(amount_field, 0), 
                    external_row.get(amount_field, 0)
                )
                
                price_match, price_diff = self._check_amount_match(
                    internal_row.get(price_field, 0), 
                    external_row.get(price_field, 0)
                )
                
                # Record the match or mismatch
                match_record = {
                    'internal_id': internal_row.get(id_field, f"internal_{i}"),
                    'external_id': external_row.get(id_field, f"external_{external_idx}"),
                    'matching_key': internal_key,
                    'internal_amount': internal_row.get(amount_field, 0),
                    'external_amount': external_row.get(amount_field, 0),
                    'internal_price': internal_row.get(price_field, 0),
                    'external_price': external_row.get(price_field, 0)
                }
                
                if amount_match and price_match:
                    # Perfect match
                    result['matched'].append(match_record)
                    internal_matched_indices.append(i)
                    external_matched_indices.append(external_idx)
                elif amount_match:
                    # Price mismatch
                    match_record['price_difference'] = price_diff
                    match_record['price_difference_pct'] = price_diff / max(
                        internal_row.get(price_field, 1), 0.000001) * 100
                    
                    if abs(price_diff) <= self.auto_resolve_threshold:
                        match_record['auto_resolved'] = True
                        match_record['resolution'] = 'auto_resolved_small_price_diff'
                        result['auto_resolved'].append(match_record)
                    else:
                        result['price_mismatches'].append(match_record)
                    
                    internal_matched_indices.append(i)
                    external_matched_indices.append(external_idx)
                elif price_match:
                    # Amount mismatch
                    match_record['amount_difference'] = amount_diff
                    match_record['amount_difference_pct'] = amount_diff / max(
                        internal_row.get(amount_field, 1), 0.000001) * 100
                    
                    if abs(amount_diff) <= self.auto_resolve_threshold:
                        match_record['auto_resolved'] = True
                        match_record['resolution'] = 'auto_resolved_small_amount_diff'
                        result['auto_resolved'].append(match_record)
                    else:
                        result['amount_mismatches'].append(match_record)
                    
                    internal_matched_indices.append(i)
                    external_matched_indices.append(external_idx)
                else:
                    # Both mismatches
                    match_record['amount_difference'] = amount_diff
                    match_record['amount_difference_pct'] = amount_diff / max(
                        internal_row.get(amount_field, 1), 0.000001) * 100
                    match_record['price_difference'] = price_diff
                    match_record['price_difference_pct'] = price_diff / max(
                        internal_row.get(price_field, 1), 0.000001) * 100
                    
                    if (abs(amount_diff) <= self.auto_resolve_threshold and 
                        abs(price_diff) <= self.auto_resolve_threshold):
                        match_record['auto_resolved'] = True
                        match_record['resolution'] = 'auto_resolved_small_diffs'
                        result['auto_resolved'].append(match_record)
                        internal_matched_indices.append(i)
                        external_matched_indices.append(external_idx)
                    else:
                        # Check which mismatch is more significant
                        if abs(amount_diff) > abs(price_diff):
                            result['amount_mismatches'].append(match_record)
                        else:
                            result['price_mismatches'].append(match_record)
                        
                        internal_matched_indices.append(i)
                        external_matched_indices.append(external_idx)
            
            elif len(matching_indices) > 1:
                # Multiple matches found - check if any are exact
                exact_match = False
                for external_idx in matching_indices:
                    external_row = df_external.loc[external_idx]
                    
                    amount_match, _ = self._check_amount_match(
                        internal_row.get(amount_field, 0), 
                        external_row.get(amount_field, 0)
                    )
                    
                    price_match, _ = self._check_amount_match(
                        internal_row.get(price_field, 0), 
                        external_row.get(price_field, 0)
                    )
                    
                    if amount_match and price_match:
                        # Found exact match among multiple
                        match_record = {
                            'internal_id': internal_row.get(id_field, f"internal_{i}"),
                            'external_id': external_row.get(id_field, f"external_{external_idx}"),
                            'matching_key': internal_key,
                            'internal_amount': internal_row.get(amount_field, 0),
                            'external_amount': external_row.get(amount_field, 0),
                            'internal_price': internal_row.get(price_field, 0),
                            'external_price': external_row.get(price_field, 0),
                            'note': 'Multiple possible matches, selected exact match'
                        }
                        
                        result['matched'].append(match_record)
                        internal_matched_indices.append(i)
                        external_matched_indices.append(external_idx)
                        exact_match = True
                        break
                
                if not exact_match:
                    # No exact match among multiple
                    unmatch_record = {
                        'internal_id': internal_row.get(id_field, f"internal_{i}"),
                        'matching_key': internal_key,
                        'internal_amount': internal_row.get(amount_field, 0),
                        'internal_price': internal_row.get(price_field, 0),
                        'potential_matches': len(matching_indices),
                        'issue': 'multiple_matches_no_exact'
                    }
                    result['unmatched_internal'].append(unmatch_record)
        
        # Add unmatched internal records
        unmatched_internal_indices = [i for i in df_internal.index if i not in internal_matched_indices]
        for i in unmatched_internal_indices:
            internal_row = df_internal.loc[i]
            unmatch_record = {
                'internal_id': internal_row.get(id_field, f"internal_{i}"),
                'matching_key': internal_row['_merge_key'],
                'internal_amount': internal_row.get(amount_field, 0),
                'internal_price': internal_row.get(price_field, 0),
                'issue': 'no_matching_external'
            }
            result['unmatched_internal'].append(unmatch_record)
        
        # Add unmatched external records
        unmatched_external_indices = [i for i in df_external.index if i not in external_matched_indices]
        for i in unmatched_external_indices:
            external_row = df_external.loc[i]
            unmatch_record = {
                'external_id': external_row.get(id_field, f"external_{i}"),
                'matching_key': external_row['_merge_key'],
                'external_amount': external_row.get(amount_field, 0),
                'external_price': external_row.get(price_field, 0),
                'issue': 'no_matching_internal'
            }
            result['unmatched_external'].append(unmatch_record)
        
        # Calculate statistics and set status
        result['statistics'] = self._calculate_statistics(result)
        result['status'] = 'completed'
        
        logger.info(f"Reconciliation completed: {len(result['matched'])} matched, "
                  f"{len(result['unmatched_internal'])} unmatched internal, "
                  f"{len(result['unmatched_external'])} unmatched external")
        
        # Store result in history
        self.reconciliation_history.append(result)
        
        return result
    
    def generate_reconciliation_report(self, reconciliation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a detailed report from reconciliation results.
        
        Args:
            reconciliation_id: Optional ID to generate report for specific reconciliation
            
        Returns:
            Report dictionary
        """
        # Find reconciliation result
        if reconciliation_id:
            recon_results = [r for r in self.reconciliation_history if r.get('reconciliation_id') == reconciliation_id]
            if not recon_results:
                logger.warning(f"No reconciliation found with ID {reconciliation_id}")
                return {'error': 'Reconciliation ID not found'}
            result = recon_results[0]
        else:
            if not self.reconciliation_history:
                logger.warning("No reconciliation history found")
                return {'error': 'No reconciliation history available'}
            result = self.reconciliation_history[-1]  # Latest result
        
        # Generate report
        report = {
            'reconciliation_id': result['reconciliation_id'],
            'timestamp': result['timestamp'],
            'summary': {
                'total_internal_trades': (
                    len(result['matched']) + 
                    len(result['unmatched_internal']) + 
                    len(result['amount_mismatches']) + 
                    len(result['price_mismatches']) +
                    len(result['auto_resolved'])
                ),
                'total_external_trades': (
                    len(result['matched']) + 
                    len(result['unmatched_external']) + 
                    len(result['amount_mismatches']) + 
                    len(result['price_mismatches']) +
                    len(result['auto_resolved'])
                ),
                'perfectly_matched': len(result['matched']),
                'auto_resolved': len(result['auto_resolved']),
                'amount_mismatches': len(result['amount_mismatches']),
                'price_mismatches': len(result['price_mismatches']),
                'unmatched_internal': len(result['unmatched_internal']),
                'unmatched_external': len(result['unmatched_external']),
            },
            'match_rate': result['statistics']['match_rate'],
            'issues_by_category': {},
            'recommended_actions': []
        }
        
        # Add issue categories
        if result['unmatched_internal']:
            report['issues_by_category']['unmatched_internal'] = len(result['unmatched_internal'])
        
        if result['unmatched_external']:
            report['issues_by_category']['unmatched_external'] = len(result['unmatched_external'])
        
        if result['amount_mismatches']:
            report['issues_by_category']['amount_mismatches'] = len(result['amount_mismatches'])
        
        if result['price_mismatches']:
            report['issues_by_category']['price_mismatches'] = len(result['price_mismatches'])
        
        # Add recommended actions
        if result['unmatched_internal']:
            report['recommended_actions'].append({
                'issue': 'unmatched_internal_trades',
                'count': len(result['unmatched_internal']),
                'action': 'Review internal trades that are missing from external system',
                'priority': 'high'
            })
        
        if result['unmatched_external']:
            report['recommended_actions'].append({
                'issue': 'unmatched_external_trades',
                'count': len(result['unmatched_external']),
                'action': 'Review external trades that are missing from internal system',
                'priority': 'high'
            })
        
        if result['amount_mismatches']:
            report['recommended_actions'].append({
                'issue': 'amount_mismatches',
                'count': len(result['amount_mismatches']),
                'action': 'Reconcile quantity differences between systems',
                'priority': 'medium'
            })
        
        if result['price_mismatches']:
            report['recommended_actions'].append({
                'issue': 'price_mismatches',
                'count': len(result['price_mismatches']),
                'action': 'Reconcile price differences between systems',
                'priority': 'medium'
            })
        
        # Add overall status
        if report['match_rate'] == 100:
            report['overall_status'] = 'perfect'
        elif report['match_rate'] >= 99:
            report['overall_status'] = 'excellent'
        elif report['match_rate'] >= 95:
            report['overall_status'] = 'good'
        elif report['match_rate'] >= 90:
            report['overall_status'] = 'acceptable'
        else:
            report['overall_status'] = 'poor'
        
        return report
    
    def suggest_corrections(self, reconciliation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Suggest corrections for reconciliation issues.
        
        Args:
            reconciliation_id: Optional ID to suggest corrections for specific reconciliation
            
        Returns:
            Corrections dictionary
        """
        # Find reconciliation result
        if reconciliation_id:
            recon_results = [r for r in self.reconciliation_history if r.get('reconciliation_id') == reconciliation_id]
            if not recon_results:
                logger.warning(f"No reconciliation found with ID {reconciliation_id}")
                return {'error': 'Reconciliation ID not found'}
            result = recon_results[0]
        else:
            if not self.reconciliation_history:
                logger.warning("No reconciliation history found")
                return {'error': 'No reconciliation history available'}
            result = self.reconciliation_history[-1]  # Latest result
        
        # Generate corrections
        corrections = {
            'reconciliation_id': result['reconciliation_id'],
            'timestamp': datetime.now().isoformat(),
            'amount_corrections': [],
            'price_corrections': [],
            'trade_additions': [],
            'trade_cancellations': []
        }
        
        # Process amount mismatches
        for mismatch in result['amount_mismatches']:
            correction = {
                'internal_id': mismatch.get('internal_id', 'unknown'),
                'external_id': mismatch.get('external_id', 'unknown'),
                'issue': 'amount_mismatch',
                'internal_amount': mismatch.get('internal_amount', 0),
                'external_amount': mismatch.get('external_amount', 0),
                'difference': mismatch.get('amount_difference', 0),
                'difference_pct': mismatch.get('amount_difference_pct', 0)
            }
            
            # Add suggested correction based on size of difference
            if abs(mismatch.get('amount_difference_pct', 0)) < 1:
                # Small difference - suggest updating internal
                correction['suggested_action'] = 'update_internal'
                correction['suggested_value'] = mismatch.get('external_amount', 0)
                correction['confidence'] = 'high'
            else:
                # Larger difference - needs manual review
                correction['suggested_action'] = 'manual_review'
                correction['confidence'] = 'low'
            
            corrections['amount_corrections'].append(correction)
        
        # Process price mismatches
        for mismatch in result['price_mismatches']:
            correction = {
                'internal_id': mismatch.get('internal_id', 'unknown'),
                'external_id': mismatch.get('external_id', 'unknown'),
                'issue': 'price_mismatch',
                'internal_price': mismatch.get('internal_price', 0),
                'external_price': mismatch.get('external_price', 0),
                'difference': mismatch.get('price_difference', 0),
                'difference_pct': mismatch.get('price_difference_pct', 0)
            }
            
            # Add suggested correction based on size of difference
            if abs(mismatch.get('price_difference_pct', 0)) < 0.5:
                # Small difference - suggest updating internal
                correction['suggested_action'] = 'update_internal'
                correction['suggested_value'] = mismatch.get('external_price', 0)
                correction['confidence'] = 'high'
            else:
                # Larger difference - needs manual review
                correction['suggested_action'] = 'manual_review'
                correction['confidence'] = 'low'
            
            corrections['price_corrections'].append(correction)
        
        # Process unmatched external trades (potential additions)
        for unmatched in result['unmatched_external']:
            correction = {
                'external_id': unmatched.get('external_id', 'unknown'),
                'issue': 'missing_from_internal',
                'external_amount': unmatched.get('external_amount', 0),
                'external_price': unmatched.get('external_price', 0),
                'suggested_action': 'add_to_internal',
                'confidence': 'medium'
            }
            corrections['trade_additions'].append(correction)
        
        # Process unmatched internal trades (potential cancellations)
        for unmatched in result['unmatched_internal']:
            correction = {
                'internal_id': unmatched.get('internal_id', 'unknown'),
                'issue': 'missing_from_external',
                'internal_amount': unmatched.get('internal_amount', 0),
                'internal_price': unmatched.get('internal_price', 0)
            }
            
            # Determine if it's a recent trade (might be in transit)
            if 'timestamp' in unmatched and isinstance(unmatched['timestamp'], str):
                try:
                    trade_time = datetime.fromisoformat(unmatched['timestamp'])
                    now = datetime.now()
                    if (now - trade_time).total_seconds() < 300:  # 5 minutes
                        correction['suggested_action'] = 'wait_for_settlement'
                        correction['confidence'] = 'high'
                    else:
                        correction['suggested_action'] = 'verify_then_cancel'
                        correction['confidence'] = 'medium'
                except (ValueError, TypeError):
                    correction['suggested_action'] = 'verify_then_cancel'
                    correction['confidence'] = 'low'
            else:
                correction['suggested_action'] = 'verify_then_cancel'
                correction['confidence'] = 'medium'
            
            corrections['trade_cancellations'].append(correction)
        
        return corrections
    
    def _create_merge_keys(self, df: pd.DataFrame, fields: List[str]) -> pd.Series:
        """Create merged keys for matching."""
        if len(fields) == 1:
            return df[fields[0]].astype(str)
        
        return df[fields].astype(str).agg('|'.join, axis=1)
    
    def _check_amount_match(self, amount1: float, amount2: float) -> Tuple[bool, float]:
        """
        Check if two amounts match within tolerance.
        
        Args:
            amount1: First amount
            amount2: Second amount
            
        Returns:
            Tuple of (is_match, difference)
        """
        difference = amount1 - amount2
        
        # Check absolute difference
        if abs(difference) <= self.tolerance_amount:
            return True, difference
        
        # Check percentage difference
        max_amount = max(abs(amount1), abs(amount2))
        if max_amount > 0:
            pct_diff = abs(difference) / max_amount
            if pct_diff <= self.tolerance_percentage:
                return True, difference
        
        return False, difference
    
    def _calculate_statistics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics from reconciliation results."""
        total_internal = (
            len(result['matched']) + 
            len(result['unmatched_internal']) + 
            len(result['amount_mismatches']) + 
            len(result['price_mismatches']) +
            len(result['auto_resolved'])
        )
        
        total_external = (
            len(result['matched']) + 
            len(result['unmatched_external']) + 
            len(result['amount_mismatches']) + 
            len(result['price_mismatches']) +
            len(result['auto_resolved'])
        )
        
        total_matched = len(result['matched']) + len(result['auto_resolved'])
        
        if total_internal > 0:
            internal_match_rate = (total_matched / total_internal) * 100
        else:
            internal_match_rate = 0
            
        if total_external > 0:
            external_match_rate = (total_matched / total_external) * 100
        else:
            external_match_rate = 0
        
        # Overall match rate is the lower of the two
        match_rate = min(internal_match_rate, external_match_rate)
        
        return {
            'total_internal': total_internal,
            'total_external': total_external,
            'total_matched': total_matched,
            'internal_match_rate': internal_match_rate,
            'external_match_rate': external_match_rate,
            'match_rate': match_rate,
            'amount_mismatches': len(result['amount_mismatches']),
            'price_mismatches': len(result['price_mismatches']),
            'auto_resolved': len(result['auto_resolved'])
        } 