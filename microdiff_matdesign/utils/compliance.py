"""Compliance and regulatory support for global deployment.

Supports GDPR, CCPA, PDPA, and other privacy regulations.
"""

import os
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta

from .logging_config import get_logger
from .error_handling import handle_errors, MicroDiffError
from .security import hash_data, generate_secure_token


class DataCategory(Enum):
    """Categories of data for compliance classification."""
    PERSONAL_IDENTIFIABLE = "pii"
    SENSITIVE_PERSONAL = "sensitive_pii"
    TECHNICAL_DATA = "technical"
    USAGE_ANALYTICS = "analytics"
    SYSTEM_LOGS = "logs"
    RESEARCH_DATA = "research"
    BUSINESS_DATA = "business"


class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


class DataRetentionPeriod(Enum):
    """Standard data retention periods."""
    SHORT_TERM = 30  # 30 days
    MEDIUM_TERM = 365  # 1 year
    LONG_TERM = 2555  # 7 years
    RESEARCH_DATA = 3650  # 10 years
    PERMANENT = -1  # Permanent retention


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    data_id: str
    data_category: DataCategory
    legal_basis: LegalBasis
    purpose: str
    processing_timestamp: float
    retention_period_days: int
    subject_id: Optional[str] = None
    consent_id: Optional[str] = None
    processor: str = "microdiff_matdesign"
    location: str = "unknown"
    
    @property
    def expiry_timestamp(self) -> float:
        """Get data expiry timestamp."""
        if self.retention_period_days == -1:
            return float('inf')  # Permanent
        return self.processing_timestamp + (self.retention_period_days * 24 * 3600)
    
    @property
    def is_expired(self) -> bool:
        """Check if data has expired."""
        return time.time() > self.expiry_timestamp


@dataclass
class ConsentRecord:
    """Record of user consent."""
    consent_id: str
    subject_id: str
    purposes: List[str]
    timestamp: float
    ip_address: str
    user_agent: str
    consent_method: str  # "explicit", "opt_in", "implied"
    withdrawn: bool = False
    withdrawal_timestamp: Optional[float] = None
    
    @property
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        return not self.withdrawn and time.time() - self.timestamp < (365 * 24 * 3600)  # 1 year


class ComplianceManager:
    """Manages compliance with privacy regulations."""
    
    def __init__(self, storage_path: str = ".compliance"):
        """Initialize compliance manager.
        
        Args:
            storage_path: Path to store compliance records
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = get_logger('compliance.manager')
        
        # In-memory caches
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_records: Dict[str, ConsentRecord] = {}
        
        # Load existing records
        self._load_records()
        
        self.logger.info("Compliance manager initialized")
    
    def _load_records(self):
        """Load compliance records from storage."""
        try:
            # Load processing records
            processing_file = self.storage_path / "processing_records.json"
            if processing_file.exists():
                with open(processing_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        record = DataProcessingRecord(**record_data)
                        # Convert enum strings back to enums
                        record.data_category = DataCategory(record_data['data_category'])
                        record.legal_basis = LegalBasis(record_data['legal_basis'])
                        self.processing_records[record.data_id] = record
            
            # Load consent records
            consent_file = self.storage_path / "consent_records.json"
            if consent_file.exists():
                with open(consent_file, 'r') as f:
                    data = json.load(f)
                    for record_data in data:
                        record = ConsentRecord(**record_data)
                        self.consent_records[record.consent_id] = record
        
        except Exception as e:
            self.logger.error(f"Failed to load compliance records: {e}")
    
    def _save_records(self):
        """Save compliance records to storage."""
        try:
            # Save processing records
            processing_file = self.storage_path / "processing_records.json"
            processing_data = []
            for record in self.processing_records.values():
                record_dict = asdict(record)
                record_dict['data_category'] = record.data_category.value
                record_dict['legal_basis'] = record.legal_basis.value
                processing_data.append(record_dict)
            
            with open(processing_file, 'w') as f:
                json.dump(processing_data, f, indent=2)
            
            # Save consent records
            consent_file = self.storage_path / "consent_records.json"
            consent_data = [asdict(record) for record in self.consent_records.values()]
            
            with open(consent_file, 'w') as f:
                json.dump(consent_data, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Failed to save compliance records: {e}")
    
    def record_data_processing(self, 
                             data_id: str,
                             data_category: DataCategory,
                             legal_basis: LegalBasis,
                             purpose: str,
                             retention_days: int,
                             subject_id: Optional[str] = None,
                             consent_id: Optional[str] = None,
                             location: str = "unknown") -> bool:
        """Record a data processing activity.
        
        Args:
            data_id: Unique identifier for the data
            data_category: Category of data being processed
            legal_basis: Legal basis for processing
            purpose: Purpose of processing
            retention_days: Data retention period in days
            subject_id: Data subject identifier
            consent_id: Consent record identifier
            location: Processing location
            
        Returns:
            True if recorded successfully
        """
        try:
            record = DataProcessingRecord(
                data_id=data_id,
                data_category=data_category,
                legal_basis=legal_basis,
                purpose=purpose,
                processing_timestamp=time.time(),
                retention_period_days=retention_days,
                subject_id=subject_id,
                consent_id=consent_id,
                location=location
            )
            
            self.processing_records[data_id] = record
            self._save_records()
            
            self.logger.info(f"Recorded data processing: {data_id}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to record data processing: {e}")
            return False
    
    def record_consent(self,
                      subject_id: str,
                      purposes: List[str],
                      ip_address: str,
                      user_agent: str,
                      consent_method: str = "explicit") -> str:
        """Record user consent.
        
        Args:
            subject_id: Data subject identifier
            purposes: List of purposes consented to
            ip_address: IP address of consent
            user_agent: User agent string
            consent_method: Method of consent collection
            
        Returns:
            Consent ID
        """
        consent_id = generate_secure_token(16)
        
        record = ConsentRecord(
            consent_id=consent_id,
            subject_id=subject_id,
            purposes=purposes,
            timestamp=time.time(),
            ip_address=ip_address,
            user_agent=user_agent,
            consent_method=consent_method
        )
        
        self.consent_records[consent_id] = record
        self._save_records()
        
        self.logger.info(f"Recorded consent: {consent_id} for subject: {subject_id}")
        return consent_id
    
    def withdraw_consent(self, consent_id: str) -> bool:
        """Withdraw user consent.
        
        Args:
            consent_id: Consent record identifier
            
        Returns:
            True if withdrawal recorded
        """
        if consent_id in self.consent_records:
            record = self.consent_records[consent_id]
            record.withdrawn = True
            record.withdrawal_timestamp = time.time()
            
            self._save_records()
            
            self.logger.info(f"Consent withdrawn: {consent_id}")
            return True
        
        return False
    
    def check_consent_validity(self, consent_id: str, purpose: str) -> bool:
        """Check if consent is valid for a specific purpose.
        
        Args:
            consent_id: Consent record identifier
            purpose: Purpose to check
            
        Returns:
            True if consent is valid
        """
        if consent_id not in self.consent_records:
            return False
        
        record = self.consent_records[consent_id]
        return record.is_valid and purpose in record.purposes
    
    def get_subject_data(self, subject_id: str) -> Dict[str, Any]:
        """Get all data for a data subject.
        
        Args:
            subject_id: Data subject identifier
            
        Returns:
            Dictionary of subject's data
        """
        subject_data = {
            'subject_id': subject_id,
            'processing_records': [],
            'consent_records': [],
            'data_categories': set()
        }
        
        # Find processing records
        for record in self.processing_records.values():
            if record.subject_id == subject_id:
                subject_data['processing_records'].append(asdict(record))
                subject_data['data_categories'].add(record.data_category.value)
        
        # Find consent records
        for record in self.consent_records.values():
            if record.subject_id == subject_id:
                subject_data['consent_records'].append(asdict(record))
        
        subject_data['data_categories'] = list(subject_data['data_categories'])
        
        return subject_data
    
    def delete_subject_data(self, subject_id: str, 
                           verify_erasure_rights: bool = True) -> Tuple[bool, List[str]]:
        """Delete all data for a data subject (GDPR Article 17).
        
        Args:
            subject_id: Data subject identifier
            verify_erasure_rights: Whether to verify legal basis for erasure
            
        Returns:
            Tuple of (success, list of errors)
        """
        errors = []
        
        try:
            if verify_erasure_rights:
                # Check if erasure is legally required/permitted
                processing_records = [r for r in self.processing_records.values() 
                                    if r.subject_id == subject_id]
                
                for record in processing_records:
                    if (record.legal_basis == LegalBasis.LEGAL_OBLIGATION or
                        record.legal_basis == LegalBasis.PUBLIC_TASK):
                        errors.append(f"Cannot erase data {record.data_id} due to legal obligation")
                    
                    if record.data_category == DataCategory.RESEARCH_DATA:
                        errors.append(f"Research data {record.data_id} may be exempt from erasure")
            
            if errors and verify_erasure_rights:
                return False, errors
            
            # Remove processing records
            to_remove = [data_id for data_id, record in self.processing_records.items()
                        if record.subject_id == subject_id]
            
            for data_id in to_remove:
                del self.processing_records[data_id]
            
            # Remove consent records
            consent_to_remove = [consent_id for consent_id, record in self.consent_records.items()
                               if record.subject_id == subject_id]
            
            for consent_id in consent_to_remove:
                del self.consent_records[consent_id]
            
            self._save_records()
            
            self.logger.info(f"Deleted all data for subject: {subject_id}")
            return True, []
        
        except Exception as e:
            self.logger.error(f"Failed to delete subject data: {e}")
            return False, [str(e)]
    
    def cleanup_expired_data(self) -> Dict[str, Any]:
        """Clean up expired data according to retention policies.
        
        Returns:
            Cleanup report
        """
        current_time = time.time()
        
        expired_records = []
        active_records = []
        
        for data_id, record in list(self.processing_records.items()):
            if record.is_expired:
                expired_records.append(data_id)
                del self.processing_records[data_id]
            else:
                active_records.append(data_id)
        
        self._save_records()
        
        report = {
            'cleanup_timestamp': current_time,
            'expired_records_removed': len(expired_records),
            'active_records_remaining': len(active_records),
            'expired_record_ids': expired_records
        }
        
        self.logger.info(f"Cleanup completed: {len(expired_records)} expired records removed")
        
        return report
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report.
        
        Returns:
            Comprehensive compliance report
        """
        current_time = time.time()
        
        # Count records by category
        category_counts = {}
        for record in self.processing_records.values():
            category = record.data_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Count records by legal basis
        legal_basis_counts = {}
        for record in self.processing_records.values():
            basis = record.legal_basis.value
            legal_basis_counts[basis] = legal_basis_counts.get(basis, 0) + 1
        
        # Count consent records
        valid_consents = sum(1 for record in self.consent_records.values() if record.is_valid)
        withdrawn_consents = sum(1 for record in self.consent_records.values() if record.withdrawn)
        
        # Find expiring data
        expiring_soon = []
        for data_id, record in self.processing_records.items():
            days_until_expiry = (record.expiry_timestamp - current_time) / (24 * 3600)
            if 0 < days_until_expiry <= 30:  # Expiring within 30 days
                expiring_soon.append({
                    'data_id': data_id,
                    'days_until_expiry': int(days_until_expiry),
                    'category': record.data_category.value
                })
        
        report = {
            'report_timestamp': current_time,
            'total_processing_records': len(self.processing_records),
            'total_consent_records': len(self.consent_records),
            'valid_consents': valid_consents,
            'withdrawn_consents': withdrawn_consents,
            'data_by_category': category_counts,
            'data_by_legal_basis': legal_basis_counts,
            'records_expiring_soon': expiring_soon,
            'compliance_score': self._calculate_compliance_score()
        }
        
        return report
    
    def _calculate_compliance_score(self) -> float:
        """Calculate compliance score based on various factors.
        
        Returns:
            Compliance score (0.0 - 1.0)
        """
        score = 1.0
        
        # Check for expired data (should be cleaned up)
        expired_count = sum(1 for record in self.processing_records.values() if record.is_expired)
        if expired_count > 0:
            score -= 0.2
        
        # Check for consent-based processing without valid consent
        consent_issues = 0
        for record in self.processing_records.values():
            if (record.legal_basis == LegalBasis.CONSENT and 
                record.consent_id and
                not self.check_consent_validity(record.consent_id, record.purpose)):
                consent_issues += 1
        
        if consent_issues > 0:
            score -= min(0.3, consent_issues * 0.1)
        
        # Check for proper legal basis documentation
        undocumented_basis = sum(1 for record in self.processing_records.values() 
                               if not record.legal_basis)
        if undocumented_basis > 0:
            score -= min(0.2, undocumented_basis * 0.05)
        
        return max(0.0, score)
    
    def anonymize_data(self, data_id: str) -> bool:
        """Anonymize data to remove personal identifiers.
        
        Args:
            data_id: Data identifier to anonymize
            
        Returns:
            True if anonymization successful
        """
        if data_id in self.processing_records:
            record = self.processing_records[data_id]
            
            # Update record to reflect anonymization
            record.subject_id = None
            record.consent_id = None
            record.data_category = DataCategory.TECHNICAL_DATA
            record.legal_basis = LegalBasis.LEGITIMATE_INTERESTS
            
            self._save_records()
            
            self.logger.info(f"Data anonymized: {data_id}")
            return True
        
        return False
    
    def export_compliance_data(self, file_path: str) -> bool:
        """Export all compliance data.
        
        Args:
            file_path: Output file path
            
        Returns:
            True if export successful
        """
        try:
            export_data = {
                'export_timestamp': time.time(),
                'processing_records': [asdict(record) for record in self.processing_records.values()],
                'consent_records': [asdict(record) for record in self.consent_records.values()],
                'compliance_report': self.generate_compliance_report()
            }
            
            # Convert enums to strings for JSON serialization
            for record in export_data['processing_records']:
                record['data_category'] = record['data_category'].value if hasattr(record['data_category'], 'value') else record['data_category']
                record['legal_basis'] = record['legal_basis'].value if hasattr(record['legal_basis'], 'value') else record['legal_basis']
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Compliance data exported to: {file_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to export compliance data: {e}")
            return False


# Global compliance manager instance
compliance_manager = ComplianceManager()

# Convenience functions
def record_processing(data_id: str, category: DataCategory, legal_basis: LegalBasis,
                     purpose: str, retention_days: int, **kwargs) -> bool:
    """Record data processing (convenience function)."""
    return compliance_manager.record_data_processing(
        data_id, category, legal_basis, purpose, retention_days, **kwargs
    )

def record_consent(subject_id: str, purposes: List[str], ip_address: str, 
                  user_agent: str, **kwargs) -> str:
    """Record consent (convenience function)."""
    return compliance_manager.record_consent(
        subject_id, purposes, ip_address, user_agent, **kwargs
    )

def check_consent(consent_id: str, purpose: str) -> bool:
    """Check consent validity (convenience function)."""
    return compliance_manager.check_consent_validity(consent_id, purpose)

def get_compliance_report() -> Dict[str, Any]:
    """Get compliance report (convenience function)."""
    return compliance_manager.generate_compliance_report()

def cleanup_expired_data() -> Dict[str, Any]:
    """Cleanup expired data (convenience function)."""
    return compliance_manager.cleanup_expired_data()


# Compliance decorators
def requires_consent(purpose: str, consent_id_param: str = 'consent_id'):
    """Decorator to require valid consent for function execution.
    
    Args:
        purpose: Purpose that requires consent
        consent_id_param: Parameter name containing consent ID
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            consent_id = kwargs.get(consent_id_param)
            
            if not consent_id or not check_consent(consent_id, purpose):
                raise MicroDiffError(f"Valid consent required for purpose: {purpose}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def track_data_processing(data_category: DataCategory, legal_basis: LegalBasis, 
                         purpose: str, retention_days: int):
    """Decorator to automatically track data processing.
    
    Args:
        data_category: Category of data being processed
        legal_basis: Legal basis for processing
        purpose: Purpose of processing
        retention_days: Data retention period
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate data ID for this processing activity
            import inspect
            func_name = func.__name__
            data_id = f"{func_name}_{hash_data(str(args) + str(kwargs))[:8]}"
            
            # Record the processing
            record_processing(data_id, data_category, legal_basis, purpose, retention_days)
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator