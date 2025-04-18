"""
Trade reporting module for regulatory and internal reporting.
Handles generation and submission of reports for trades.
"""
import logging
import json
import csv
import os
from io import StringIO
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
import xml.dom.minidom

logger = logging.getLogger(__name__)

class TradeReporting:
    """
    Trade reporting system for generating regulatory and internal reports.
    """
    
    def __init__(self, output_dir: str = './reports'):
        """
        Initialize the trade reporting system.
        
        Args:
            output_dir: Directory for storing generated reports
        """
        self.output_dir = output_dir
        self.report_formats = {
            'csv': self._generate_csv_report,
            'json': self._generate_json_report,
            'xml': self._generate_xml_report,
            'text': self._generate_text_report
        }
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_regulatory_report(self, 
                                  trades: List[Dict[str, Any]],
                                  report_type: str,
                                  start_date: Union[str, datetime],
                                  end_date: Union[str, datetime],
                                  format: str = 'csv',
                                  save_file: bool = True) -> Dict[str, Any]:
        """
        Generate a regulatory report for specified trade data.
        
        Args:
            trades: List of trade data dictionaries
            report_type: Type of regulatory report ('mifid', 'emir', 'finra', etc.)
            start_date: Start date for the report period
            end_date: End date for the report period
            format: Output format (csv, json, xml, text)
            save_file: Whether to save the report to a file
            
        Returns:
            Report information with content
        """
        logger.info(f"Generating {report_type} regulatory report from {start_date} to {end_date}")
        
        # Handle date formatting
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)
        
        # Filter trades by date range
        filtered_trades = self._filter_trades_by_date(trades, start_date, end_date)
        
        # Create report info
        report_info = {
            'report_id': f"{report_type}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}",
            'report_type': report_type,
            'generation_time': datetime.now().isoformat(),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'trade_count': len(filtered_trades),
            'format': format,
            'content': None,
            'file_path': None
        }
        
        # Process trades according to regulatory needs
        processed_trades = self._process_for_regulatory(filtered_trades, report_type)
        
        # Generate report in requested format
        if format in self.report_formats:
            report_content = self.report_formats[format](processed_trades, report_info)
            report_info['content'] = report_content
        else:
            logger.error(f"Unsupported format: {format}")
            report_info['error'] = f"Unsupported format: {format}"
            return report_info
        
        # Save to file if requested
        if save_file:
            file_name = f"{report_info['report_id']}.{format}"
            file_path = os.path.join(self.output_dir, file_name)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                report_info['file_path'] = file_path
                logger.info(f"Regulatory report saved to {file_path}")
            except Exception as e:
                logger.error(f"Error saving report to file: {str(e)}")
                report_info['error'] = f"Error saving report: {str(e)}"
        
        return report_info
    
    def generate_internal_report(self, 
                                trades: List[Dict[str, Any]],
                                report_name: str,
                                parameters: Dict[str, Any] = None,
                                format: str = 'csv',
                                save_file: bool = True) -> Dict[str, Any]:
        """
        Generate an internal report for analysis or operations.
        
        Args:
            trades: List of trade data dictionaries
            report_name: Name of the internal report
            parameters: Additional parameters for report customization
            format: Output format (csv, json, xml, text)
            save_file: Whether to save the report to a file
            
        Returns:
            Report information with content
        """
        logger.info(f"Generating internal {report_name} report")
        
        parameters = parameters or {}
        
        # Create report info
        report_info = {
            'report_id': f"internal_{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'report_name': report_name,
            'generation_time': datetime.now().isoformat(),
            'parameters': parameters,
            'trade_count': len(trades),
            'format': format,
            'content': None,
            'file_path': None
        }
        
        # Process trades according to internal report needs
        processed_trades = self._process_for_internal(trades, report_name, parameters)
        
        # Generate report in requested format
        if format in self.report_formats:
            report_content = self.report_formats[format](processed_trades, report_info)
            report_info['content'] = report_content
        else:
            logger.error(f"Unsupported format: {format}")
            report_info['error'] = f"Unsupported format: {format}"
            return report_info
        
        # Save to file if requested
        if save_file:
            file_name = f"{report_info['report_id']}.{format}"
            file_path = os.path.join(self.output_dir, file_name)
            
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                report_info['file_path'] = file_path
                logger.info(f"Internal report saved to {file_path}")
            except Exception as e:
                logger.error(f"Error saving report to file: {str(e)}")
                report_info['error'] = f"Error saving report: {str(e)}"
        
        return report_info
    
    def _filter_trades_by_date(self, 
                              trades: List[Dict[str, Any]], 
                              start_date: datetime, 
                              end_date: datetime) -> List[Dict[str, Any]]:
        """Filter trades to include only those within the date range."""
        filtered = []
        
        for trade in trades:
            trade_time = None
            
            # Look for timestamp field in common formats
            for field in ['timestamp', 'trade_time', 'execution_time', 'time']:
                if field in trade:
                    try:
                        if isinstance(trade[field], str):
                            trade_time = datetime.fromisoformat(trade[field])
                        else:
                            trade_time = trade[field]
                        break
                    except (ValueError, TypeError):
                        continue
            
            # Skip trades with no valid timestamp
            if not trade_time:
                continue
            
            # Include if within range
            if start_date <= trade_time <= end_date:
                filtered.append(trade)
        
        return filtered
    
    def _process_for_regulatory(self, 
                               trades: List[Dict[str, Any]], 
                               report_type: str) -> List[Dict[str, Any]]:
        """Process trades according to regulatory reporting requirements."""
        processed = []
        
        for trade in trades:
            # Clone the trade data
            processed_trade = trade.copy()
            
            # Apply specific transformations based on report type
            if report_type.lower() == 'mifid':
                # MiFID II reporting fields
                processed_trade = self._transform_for_mifid(processed_trade)
            elif report_type.lower() == 'emir':
                # EMIR reporting fields
                processed_trade = self._transform_for_emir(processed_trade)
            elif report_type.lower() == 'finra':
                # FINRA reporting fields
                processed_trade = self._transform_for_finra(processed_trade)
            else:
                # Default minimal processing
                processed_trade = self._standardize_trade_data(processed_trade)
            
            processed.append(processed_trade)
        
        return processed
    
    def _process_for_internal(self, 
                             trades: List[Dict[str, Any]], 
                             report_name: str,
                             parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process trades according to internal reporting needs."""
        processed = []
        
        for trade in trades:
            # Clone the trade data
            processed_trade = trade.copy()
            
            # Apply specific transformations based on report name
            if report_name.lower() == 'daily_summary':
                # Daily trading summary
                processed_trade = self._transform_for_daily_summary(processed_trade)
            elif report_name.lower() == 'performance_analysis':
                # Trading performance analysis
                processed_trade = self._transform_for_performance(processed_trade, parameters)
            elif report_name.lower() == 'risk_report':
                # Risk reporting
                processed_trade = self._transform_for_risk(processed_trade, parameters)
            else:
                # Default minimal processing
                processed_trade = self._standardize_trade_data(processed_trade)
            
            processed.append(processed_trade)
        
        return processed
    
    def _transform_for_mifid(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trade data for MiFID II reporting."""
        mifid_trade = {
            'transaction_reference_number': trade.get('trade_id', 'unknown'),
            'trading_venue_transaction_id': trade.get('venue_trade_id', ''),
            'trading_date_time': self._format_datetime(trade.get('timestamp', '')),
            'trading_capacity': trade.get('capacity', 'DEAL'),  # DEAL, MTCH, AOTC
            'quantity': trade.get('quantity', 0),
            'quantity_currency': trade.get('quantity_currency', ''),
            'price': trade.get('price', 0),
            'price_currency': trade.get('price_currency', 'USD'),
            'venue': trade.get('venue', ''),
            'instrument_id': trade.get('instrument_id', ''),
            'buyer_id': trade.get('buyer_id', ''),
            'seller_id': trade.get('seller_id', ''),
            'country_of_branch': trade.get('country_of_branch', ''),
            'up_front_payment': trade.get('up_front_payment', 0),
            'up_front_payment_currency': trade.get('up_front_payment_currency', ''),
            'complex_trade_component_id': trade.get('complex_trade_component_id', '')
        }
        return mifid_trade
    
    def _transform_for_emir(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trade data for EMIR reporting."""
        emir_trade = {
            'trade_id': trade.get('trade_id', 'unknown'),
            'reporting_timestamp': self._format_datetime(datetime.now()),
            'counterparty_id': trade.get('counterparty_id', ''),
            'counterparty_side': 'B' if trade.get('side', '').lower() == 'sell' else 'S',
            'trade_date': self._format_date(trade.get('timestamp', '')),
            'value_date': self._format_date(trade.get('settlement_date', '')),
            'contract_type': trade.get('contract_type', ''),
            'product_id1': trade.get('product_id', ''),
            'product_id2': trade.get('underlying_id', ''),
            'quantity': trade.get('quantity', 0),
            'price': trade.get('price', 0),
            'notional_amount': trade.get('notional_amount', 0),
            'currency': trade.get('currency', 'USD'),
            'clearing_status': trade.get('clearing_status', 'U'),  # C=cleared, U=uncleared
            'execution_venue': trade.get('venue', '')
        }
        return emir_trade
    
    def _transform_for_finra(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trade data for FINRA reporting."""
        finra_trade = {
            'trade_report_id': trade.get('trade_id', 'unknown'),
            'execution_date': self._format_date(trade.get('timestamp', '')),
            'execution_time': self._format_time(trade.get('timestamp', '')),
            'security_id': trade.get('symbol', ''),
            'security_id_type': trade.get('symbol_type', 'S'),  # S=Symbol, C=CUSIP
            'quantity': trade.get('quantity', 0),
            'price': trade.get('price', 0),
            'reporting_side': 'B' if trade.get('side', '').lower() == 'buy' else 'S',
            'contra_side': 'S' if trade.get('side', '').lower() == 'buy' else 'B',
            'capacity': trade.get('capacity', 'P'),  # P=Principal, A=Agency
            'trading_market_center_id': trade.get('venue', ''),
            'settlement_date': self._format_date(trade.get('settlement_date', '')),
            'transaction_id': trade.get('transaction_id', ''),
            'memo': trade.get('memo', '')
        }
        return finra_trade
    
    def _transform_for_daily_summary(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trade data for daily summary reporting."""
        summary_trade = {
            'trade_id': trade.get('trade_id', 'unknown'),
            'trade_date': self._format_date(trade.get('timestamp', '')),
            'symbol': trade.get('symbol', ''),
            'side': trade.get('side', ''),
            'quantity': trade.get('quantity', 0),
            'price': trade.get('price', 0),
            'notional_value': trade.get('quantity', 0) * trade.get('price', 0),
            'venue': trade.get('venue', ''),
            'trader': trade.get('trader', ''),
            'strategy': trade.get('strategy', ''),
            'commission': trade.get('commission', 0),
            'status': trade.get('status', '')
        }
        return summary_trade
    
    def _transform_for_performance(self, trade: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trade data for performance reporting."""
        benchmark_price = parameters.get('benchmark_price', {}).get(trade.get('symbol', ''), 0)
        
        performance_trade = {
            'trade_id': trade.get('trade_id', 'unknown'),
            'trade_date': self._format_date(trade.get('timestamp', '')),
            'symbol': trade.get('symbol', ''),
            'side': trade.get('side', ''),
            'quantity': trade.get('quantity', 0),
            'price': trade.get('price', 0),
            'notional_value': trade.get('quantity', 0) * trade.get('price', 0),
            'benchmark_price': benchmark_price,
            'price_improvement': trade.get('price', 0) - benchmark_price if benchmark_price > 0 else 0,
            'strategy': trade.get('strategy', ''),
            'algorithm': trade.get('algorithm', ''),
            'execution_time_ms': trade.get('execution_time_ms', 0)
        }
        return performance_trade
    
    def _transform_for_risk(self, trade: Dict[str, Any], parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trade data for risk reporting."""
        risk_trade = {
            'trade_id': trade.get('trade_id', 'unknown'),
            'trade_date': self._format_date(trade.get('timestamp', '')),
            'symbol': trade.get('symbol', ''),
            'side': trade.get('side', ''),
            'quantity': trade.get('quantity', 0),
            'price': trade.get('price', 0),
            'notional_value': trade.get('quantity', 0) * trade.get('price', 0),
            'trader': trade.get('trader', ''),
            'trading_desk': trade.get('trading_desk', ''),
            'risk_limit_usage': trade.get('risk_limit_usage', 0),
            'var_contribution': trade.get('var_contribution', 0),
            'stress_test_impact': trade.get('stress_test_impact', 0),
            'risk_factor_exposures': trade.get('risk_factor_exposures', {})
        }
        return risk_trade
    
    def _standardize_trade_data(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize trade data for general reporting."""
        standard_trade = {
            'trade_id': trade.get('trade_id', 'unknown'),
            'timestamp': self._format_datetime(trade.get('timestamp', '')),
            'symbol': trade.get('symbol', ''),
            'side': trade.get('side', ''),
            'quantity': trade.get('quantity', 0),
            'price': trade.get('price', 0),
            'venue': trade.get('venue', ''),
            'trader': trade.get('trader', ''),
            'status': trade.get('status', '')
        }
        return standard_trade
    
    def _generate_csv_report(self, data: List[Dict[str, Any]], report_info: Dict[str, Any]) -> str:
        """Generate a CSV format report."""
        if not data:
            return "No data available for report"
        
        output = StringIO()
        fieldnames = data[0].keys()
        
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        
        return output.getvalue()
    
    def _generate_json_report(self, data: List[Dict[str, Any]], report_info: Dict[str, Any]) -> str:
        """Generate a JSON format report."""
        report = {
            'report_info': {k: v for k, v in report_info.items() if k != 'content'},
            'data': data
        }
        return json.dumps(report, indent=2)
    
    def _generate_xml_report(self, data: List[Dict[str, Any]], report_info: Dict[str, Any]) -> str:
        """Generate an XML format report."""
        root = ET.Element('Report')
        
        # Add report info
        info_elem = ET.SubElement(root, 'ReportInfo')
        for key, value in report_info.items():
            if key != 'content':
                ET.SubElement(info_elem, key).text = str(value)
        
        # Add data records
        data_elem = ET.SubElement(root, 'Data')
        for item in data:
            record_elem = ET.SubElement(data_elem, 'Record')
            for key, value in item.items():
                ET.SubElement(record_elem, key).text = str(value)
        
        # Convert to pretty XML string
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = xml.dom.minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")
    
    def _generate_text_report(self, data: List[Dict[str, Any]], report_info: Dict[str, Any]) -> str:
        """Generate a plain text format report."""
        output = []
        
        # Add report header
        output.append("=" * 80)
        output.append(f"REPORT: {report_info.get('report_name', report_info.get('report_type', 'Unknown'))}")
        output.append(f"ID: {report_info.get('report_id', 'Unknown')}")
        output.append(f"Generated: {report_info.get('generation_time', 'Unknown')}")
        output.append("=" * 80)
        output.append("")
        
        # Add report parameters if available
        if 'parameters' in report_info and report_info['parameters']:
            output.append("PARAMETERS:")
            for key, value in report_info['parameters'].items():
                output.append(f"  {key}: {value}")
            output.append("")
        
        # Add data counts
        if 'trade_count' in report_info:
            output.append(f"Total trades: {report_info['trade_count']}")
        elif 'execution_count' in report_info:
            output.append(f"Total executions: {report_info['execution_count']}")
        output.append("")
        
        # Add data records
        if data:
            # Column headers
            headers = list(data[0].keys())
            col_widths = {header: max(len(header), max(len(str(item.get(header, ''))) for item in data)) for header in headers}
            
            # Header row
            header_row = " | ".join(header.ljust(col_widths[header]) for header in headers)
            output.append(header_row)
            output.append("-" * len(header_row))
            
            # Data rows
            for item in data:
                row = " | ".join(str(item.get(header, '')).ljust(col_widths[header]) for header in headers)
                output.append(row)
        else:
            output.append("No data available for report")
        
        return "\n".join(output)
    
    def _format_datetime(self, dt_value) -> str:
        """Format datetime value to ISO format string."""
        if isinstance(dt_value, datetime):
            return dt_value.isoformat()
        elif isinstance(dt_value, str):
            try:
                return datetime.fromisoformat(dt_value).isoformat()
            except ValueError:
                return dt_value
        return str(dt_value)
    
    def _format_date(self, dt_value) -> str:
        """Format date value to YYYY-MM-DD string."""
        if isinstance(dt_value, datetime):
            return dt_value.strftime('%Y-%m-%d')
        elif isinstance(dt_value, str):
            try:
                return datetime.fromisoformat(dt_value).strftime('%Y-%m-%d')
            except ValueError:
                return dt_value
        return str(dt_value)
    
    def _format_time(self, dt_value) -> str:
        """Format time value to HH:MM:SS string."""
        if isinstance(dt_value, datetime):
            return dt_value.strftime('%H:%M:%S')
        elif isinstance(dt_value, str):
            try:
                return datetime.fromisoformat(dt_value).strftime('%H:%M:%S')
            except ValueError:
                return dt_value
        return str(dt_value) 