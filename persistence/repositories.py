"""
Repository classes for database operations.

Provides a clean interface for CRUD operations on evaluation results,
custom rules, configurations, and audit logs.
"""

import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy import func, desc, and_, or_
from sqlalchemy.orm import Session

from .database import (
    Database, 
    EvaluationRecord, 
    RuleRecord, 
    ConfigRecord, 
    AuditLog,
    MetricsSummary,
    get_database
)


class EvaluationRepository:
    """
    Repository for managing evaluation records.
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()
    
    def _hash_text(self, text: str) -> str:
        """Create hash of text for deduplication."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def save(
        self,
        evaluation_id: str,
        prompt: str,
        response: str,
        passed: bool,
        scores: Dict[str, float],
        flags: List[str],
        metadata: Optional[Dict] = None,
        detailed_results: Optional[Dict] = None,
        latency_ms: float = 0,
        source: str = 'api',
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> EvaluationRecord:
        """
        Save an evaluation record.
        
        Args:
            evaluation_id: Unique evaluation identifier
            prompt: Input prompt
            response: LLM response
            passed: Overall pass/fail status
            scores: Dictionary of metric scores
            flags: List of triggered flags
            metadata: Additional metadata
            detailed_results: Detailed evaluator results
            latency_ms: Evaluation latency
            source: Source of evaluation (api, cli, dashboard)
            user_id: User identifier if available
            session_id: Session identifier
            
        Returns:
            Saved EvaluationRecord
        """
        with self.db.session() as session:
            record = EvaluationRecord(
                evaluation_id=evaluation_id,
                prompt=prompt,
                response=response,
                prompt_hash=self._hash_text(prompt),
                response_hash=self._hash_text(response),
                passed=passed,
                scores=scores,
                flags=flags,
                metadata=metadata or {},
                detailed_results=detailed_results or {},
                latency_ms=latency_ms,
                source=source,
                user_id=user_id,
                session_id=session_id
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
    
    def get_by_id(self, evaluation_id: str) -> Optional[EvaluationRecord]:
        """Get evaluation by ID."""
        with self.db.session() as session:
            return session.query(EvaluationRecord).filter(
                EvaluationRecord.evaluation_id == evaluation_id
            ).first()
    
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        passed: Optional[bool] = None,
        source: Optional[str] = None,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        order_by: str = 'timestamp',
        order_desc: bool = True
    ) -> Tuple[List[EvaluationRecord], int]:
        """
        List evaluations with filtering and pagination.
        
        Returns:
            Tuple of (records, total_count)
        """
        with self.db.session() as session:
            query = session.query(EvaluationRecord)
            
            # Apply filters
            if passed is not None:
                query = query.filter(EvaluationRecord.passed == passed)
            if source:
                query = query.filter(EvaluationRecord.source == source)
            if user_id:
                query = query.filter(EvaluationRecord.user_id == user_id)
            if start_date:
                query = query.filter(EvaluationRecord.timestamp >= start_date)
            if end_date:
                query = query.filter(EvaluationRecord.timestamp <= end_date)
            
            # Get total count
            total = query.count()
            
            # Apply ordering
            order_column = getattr(EvaluationRecord, order_by, EvaluationRecord.timestamp)
            if order_desc:
                query = query.order_by(desc(order_column))
            else:
                query = query.order_by(order_column)
            
            # Apply pagination
            records = query.offset(offset).limit(limit).all()
            
            return records, total
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get aggregated statistics for evaluations.
        """
        with self.db.session() as session:
            query = session.query(EvaluationRecord)
            
            if start_date:
                query = query.filter(EvaluationRecord.timestamp >= start_date)
            if end_date:
                query = query.filter(EvaluationRecord.timestamp <= end_date)
            
            total = query.count()
            passed = query.filter(EvaluationRecord.passed == True).count()
            failed = total - passed
            
            # Average latency
            avg_latency = session.query(
                func.avg(EvaluationRecord.latency_ms)
            ).filter(
                EvaluationRecord.timestamp >= start_date if start_date else True,
                EvaluationRecord.timestamp <= end_date if end_date else True
            ).scalar() or 0
            
            return {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": (passed / total * 100) if total > 0 else 0,
                "avg_latency_ms": float(avg_latency)
            }
    
    def delete_old(self, days: int = 90) -> int:
        """
        Delete evaluations older than specified days.
        
        Returns:
            Number of deleted records
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self.db.session() as session:
            count = session.query(EvaluationRecord).filter(
                EvaluationRecord.timestamp < cutoff
            ).delete()
            return count
    
    def search(
        self,
        query_text: str,
        limit: int = 50
    ) -> List[EvaluationRecord]:
        """
        Search evaluations by prompt or response content.
        """
        with self.db.session() as session:
            return session.query(EvaluationRecord).filter(
                or_(
                    EvaluationRecord.prompt.ilike(f"%{query_text}%"),
                    EvaluationRecord.response.ilike(f"%{query_text}%")
                )
            ).order_by(desc(EvaluationRecord.timestamp)).limit(limit).all()


class RuleRepository:
    """
    Repository for managing custom rules.
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()
    
    def save(
        self,
        rule_id: str,
        name: str,
        rule_type: str,
        pattern: str,
        severity: float = 0.5,
        action: str = 'flag',
        apply_to: str = 'both',
        description: Optional[str] = None,
        priority: int = 0
    ) -> RuleRecord:
        """
        Save a custom rule.
        """
        with self.db.session() as session:
            # Check if rule exists
            existing = session.query(RuleRecord).filter(
                RuleRecord.rule_id == rule_id
            ).first()
            
            if existing:
                # Update existing
                existing.name = name
                existing.rule_type = rule_type
                existing.pattern = pattern
                existing.severity = severity
                existing.action = action
                existing.apply_to = apply_to
                existing.description = description
                existing.priority = priority
                session.commit()
                session.refresh(existing)
                return existing
            else:
                # Create new
                record = RuleRecord(
                    rule_id=rule_id,
                    name=name,
                    rule_type=rule_type,
                    pattern=pattern,
                    severity=severity,
                    action=action,
                    apply_to=apply_to,
                    description=description,
                    priority=priority
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                return record
    
    def get_by_id(self, rule_id: str) -> Optional[RuleRecord]:
        """Get rule by ID."""
        with self.db.session() as session:
            return session.query(RuleRecord).filter(
                RuleRecord.rule_id == rule_id
            ).first()
    
    def list(
        self,
        enabled_only: bool = False,
        apply_to: Optional[str] = None
    ) -> List[RuleRecord]:
        """List all rules."""
        with self.db.session() as session:
            query = session.query(RuleRecord)
            
            if enabled_only:
                query = query.filter(RuleRecord.enabled == True)
            if apply_to:
                query = query.filter(
                    or_(
                        RuleRecord.apply_to == apply_to,
                        RuleRecord.apply_to == 'both'
                    )
                )
            
            return query.order_by(desc(RuleRecord.priority)).all()
    
    def delete(self, rule_id: str) -> bool:
        """Delete a rule."""
        with self.db.session() as session:
            count = session.query(RuleRecord).filter(
                RuleRecord.rule_id == rule_id
            ).delete()
            return count > 0
    
    def toggle(self, rule_id: str, enabled: bool) -> Optional[RuleRecord]:
        """Enable or disable a rule."""
        with self.db.session() as session:
            rule = session.query(RuleRecord).filter(
                RuleRecord.rule_id == rule_id
            ).first()
            
            if rule:
                rule.enabled = enabled
                session.commit()
                session.refresh(rule)
            
            return rule
    
    def increment_match_count(self, rule_id: str):
        """Increment the match count for a rule."""
        with self.db.session() as session:
            rule = session.query(RuleRecord).filter(
                RuleRecord.rule_id == rule_id
            ).first()
            
            if rule:
                rule.match_count += 1
                rule.last_matched_at = datetime.utcnow()


class ConfigRepository:
    """
    Repository for managing configurations.
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()
    
    def save(
        self,
        name: str,
        config_data: Dict[str, Any],
        description: Optional[str] = None,
        activate: bool = False
    ) -> ConfigRecord:
        """
        Save a configuration.
        """
        with self.db.session() as session:
            # Deactivate all if activating this one
            if activate:
                session.query(ConfigRecord).update({ConfigRecord.is_active: False})
            
            # Check if exists
            existing = session.query(ConfigRecord).filter(
                ConfigRecord.name == name
            ).first()
            
            if existing:
                existing.config_data = config_data
                existing.description = description
                existing.is_active = activate
                session.commit()
                session.refresh(existing)
                return existing
            else:
                record = ConfigRecord(
                    name=name,
                    config_data=config_data,
                    description=description,
                    is_active=activate
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                return record
    
    def get_active(self) -> Optional[ConfigRecord]:
        """Get the active configuration."""
        with self.db.session() as session:
            return session.query(ConfigRecord).filter(
                ConfigRecord.is_active == True
            ).first()
    
    def get_by_name(self, name: str) -> Optional[ConfigRecord]:
        """Get configuration by name."""
        with self.db.session() as session:
            return session.query(ConfigRecord).filter(
                ConfigRecord.name == name
            ).first()
    
    def list(self) -> List[ConfigRecord]:
        """List all configurations."""
        with self.db.session() as session:
            return session.query(ConfigRecord).order_by(
                desc(ConfigRecord.updated_at)
            ).all()
    
    def activate(self, name: str) -> Optional[ConfigRecord]:
        """Activate a configuration."""
        with self.db.session() as session:
            # Deactivate all
            session.query(ConfigRecord).update({ConfigRecord.is_active: False})
            
            # Activate the specified one
            config = session.query(ConfigRecord).filter(
                ConfigRecord.name == name
            ).first()
            
            if config:
                config.is_active = True
                session.commit()
                session.refresh(config)
            
            return config
    
    def delete(self, name: str) -> bool:
        """Delete a configuration."""
        with self.db.session() as session:
            count = session.query(ConfigRecord).filter(
                ConfigRecord.name == name
            ).delete()
            return count > 0


class AuditRepository:
    """
    Repository for audit logging.
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()
    
    def log(
        self,
        event_type: str,
        event_action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        old_value: Optional[Dict] = None,
        new_value: Optional[Dict] = None,
        metadata: Optional[Dict] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> AuditLog:
        """
        Create an audit log entry.
        """
        with self.db.session() as session:
            record = AuditLog(
                event_type=event_type,
                event_action=event_action,
                resource_type=resource_type,
                resource_id=resource_id,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                old_value=old_value,
                new_value=new_value,
                metadata=metadata or {},
                success=success,
                error_message=error_message
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record
    
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[AuditLog], int]:
        """
        List audit logs with filtering.
        """
        with self.db.session() as session:
            query = session.query(AuditLog)
            
            if event_type:
                query = query.filter(AuditLog.event_type == event_type)
            if user_id:
                query = query.filter(AuditLog.user_id == user_id)
            if resource_type:
                query = query.filter(AuditLog.resource_type == resource_type)
            if start_date:
                query = query.filter(AuditLog.timestamp >= start_date)
            if end_date:
                query = query.filter(AuditLog.timestamp <= end_date)
            
            total = query.count()
            records = query.order_by(desc(AuditLog.timestamp)).offset(offset).limit(limit).all()
            
            return records, total
    
    def delete_old(self, days: int = 365) -> int:
        """Delete old audit logs."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        with self.db.session() as session:
            count = session.query(AuditLog).filter(
                AuditLog.timestamp < cutoff
            ).delete()
            return count


class MetricsRepository:
    """
    Repository for aggregated metrics.
    """
    
    def __init__(self, db: Optional[Database] = None):
        self.db = db or get_database()
    
    def aggregate_hourly(self, hour: datetime) -> MetricsSummary:
        """
        Aggregate metrics for a specific hour.
        """
        period_start = hour.replace(minute=0, second=0, microsecond=0)
        period_end = period_start + timedelta(hours=1)
        
        with self.db.session() as session:
            # Get evaluation stats
            evals = session.query(EvaluationRecord).filter(
                EvaluationRecord.timestamp >= period_start,
                EvaluationRecord.timestamp < period_end
            ).all()
            
            total = len(evals)
            passed = sum(1 for e in evals if e.passed)
            
            # Calculate averages
            latencies = [e.latency_ms for e in evals if e.latency_ms]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            # Score averages
            toxicity_scores = [e.scores.get('toxicity', 0) for e in evals if e.scores]
            alignment_scores = [e.scores.get('alignment', 0) for e in evals if e.scores]
            
            # Count flags
            flag_counts = {}
            for e in evals:
                for flag in e.flags:
                    flag_type = flag.split(':')[0] if ':' in flag else flag[:30]
                    flag_counts[flag_type] = flag_counts.get(flag_type, 0) + 1
            
            # Check if already exists
            existing = session.query(MetricsSummary).filter(
                MetricsSummary.period_start == period_start,
                MetricsSummary.period_type == 'hourly'
            ).first()
            
            if existing:
                summary = existing
                summary.total_evaluations = total
                summary.passed_count = passed
                summary.failed_count = total - passed
                summary.avg_latency_ms = avg_latency
                summary.flag_counts = flag_counts
            else:
                summary = MetricsSummary(
                    period_start=period_start,
                    period_end=period_end,
                    period_type='hourly',
                    total_evaluations=total,
                    passed_count=passed,
                    failed_count=total - passed,
                    avg_toxicity=sum(toxicity_scores) / len(toxicity_scores) if toxicity_scores else None,
                    avg_alignment=sum(alignment_scores) / len(alignment_scores) if alignment_scores else None,
                    avg_latency_ms=avg_latency,
                    flag_counts=flag_counts
                )
                session.add(summary)
            
            session.commit()
            session.refresh(summary)
            return summary
    
    def get_dashboard_data(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get dashboard data for the last N hours.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with self.db.session() as session:
            summaries = session.query(MetricsSummary).filter(
                MetricsSummary.period_start >= cutoff,
                MetricsSummary.period_type == 'hourly'
            ).order_by(MetricsSummary.period_start).all()
            
            return {
                "periods": [s.period_start.isoformat() for s in summaries],
                "total_evaluations": [s.total_evaluations for s in summaries],
                "passed_count": [s.passed_count for s in summaries],
                "failed_count": [s.failed_count for s in summaries],
                "avg_latency": [s.avg_latency_ms for s in summaries]
            }
