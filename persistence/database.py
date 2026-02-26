"""
Database models and connection management for LLM Guardrail Studio.

Supports SQLite (default) and PostgreSQL for production use.
Uses SQLAlchemy ORM for database operations.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, 
    Text, DateTime, JSON, ForeignKey, Index, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class EvaluationRecord(Base):
    """
    Stores evaluation results for audit and analytics.
    """
    __tablename__ = 'evaluations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Input data
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    prompt_hash = Column(String(64), index=True)  # For deduplication
    response_hash = Column(String(64), index=True)
    
    # Results
    passed = Column(Boolean, default=True, index=True)
    scores = Column(JSON, default={})
    flags = Column(JSON, default=[])
    metadata = Column(JSON, default={})
    detailed_results = Column(JSON, default={})
    
    # Performance
    latency_ms = Column(Float, default=0)
    
    # Source tracking
    source = Column(String(50), default='api')  # api, cli, dashboard
    user_id = Column(String(100), nullable=True, index=True)
    session_id = Column(String(100), nullable=True)
    
    # Indexes for common queries
    __table_args__ = (
        Index('ix_evaluations_timestamp_passed', 'timestamp', 'passed'),
        Index('ix_evaluations_source_timestamp', 'source', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "evaluation_id": self.evaluation_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "prompt": self.prompt,
            "response": self.response,
            "passed": self.passed,
            "scores": self.scores,
            "flags": self.flags,
            "metadata": self.metadata,
            "detailed_results": self.detailed_results,
            "latency_ms": self.latency_ms,
            "source": self.source,
            "user_id": self.user_id
        }


class RuleRecord(Base):
    """
    Stores custom filtering rules.
    """
    __tablename__ = 'rules'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String(100), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Rule definition
    rule_type = Column(String(20), nullable=False)  # regex, keyword, script
    pattern = Column(Text, nullable=False)
    severity = Column(Float, default=0.5)
    action = Column(String(20), default='flag')  # flag, block, score
    apply_to = Column(String(20), default='both')  # prompt, response, both
    
    # Status
    enabled = Column(Boolean, default=True)
    priority = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Statistics
    match_count = Column(Integer, default=0)
    last_matched_at = Column(DateTime, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "type": self.rule_type,
            "pattern": self.pattern,
            "severity": self.severity,
            "action": self.action,
            "apply_to": self.apply_to,
            "enabled": self.enabled,
            "priority": self.priority,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "match_count": self.match_count
        }


class ConfigRecord(Base):
    """
    Stores pipeline configurations.
    """
    __tablename__ = 'configs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    # Configuration data
    config_data = Column(JSON, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "config_data": self.config_data,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class AuditLog(Base):
    """
    Audit log for tracking changes and actions.
    """
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Event information
    event_type = Column(String(100), nullable=False, index=True)
    event_action = Column(String(50), nullable=False)  # create, update, delete, evaluate
    
    # Resource information
    resource_type = Column(String(50), nullable=True)  # rule, config, evaluation
    resource_id = Column(String(100), nullable=True)
    
    # Actor information
    user_id = Column(String(100), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Event data
    old_value = Column(JSON, nullable=True)
    new_value = Column(JSON, nullable=True)
    metadata = Column(JSON, default={})
    
    # Status
    success = Column(Boolean, default=True)
    error_message = Column(Text, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "event_action": self.event_action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "user_id": self.user_id,
            "success": self.success,
            "error_message": self.error_message
        }


class MetricsSummary(Base):
    """
    Stores aggregated metrics for dashboards.
    """
    __tablename__ = 'metrics_summary'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    period_type = Column(String(20), nullable=False)  # hourly, daily, weekly
    
    # Counts
    total_evaluations = Column(Integer, default=0)
    passed_count = Column(Integer, default=0)
    failed_count = Column(Integer, default=0)
    
    # Score averages
    avg_toxicity = Column(Float, nullable=True)
    avg_alignment = Column(Float, nullable=True)
    avg_hallucination = Column(Float, nullable=True)
    avg_pii_risk = Column(Float, nullable=True)
    
    # Latency
    avg_latency_ms = Column(Float, default=0)
    p95_latency_ms = Column(Float, default=0)
    p99_latency_ms = Column(Float, default=0)
    
    # Flag counts by type
    flag_counts = Column(JSON, default={})
    
    __table_args__ = (
        Index('ix_metrics_period', 'period_type', 'period_start'),
    )


class Database:
    """
    Database connection manager with session handling.
    """
    
    def __init__(
        self, 
        connection_string: Optional[str] = None,
        echo: bool = False
    ):
        """
        Initialize database connection.
        
        Args:
            connection_string: SQLAlchemy connection string
                             Default: sqlite:///guardrails.db
            echo: Enable SQL query logging
        """
        self.connection_string = connection_string or os.getenv(
            'GUARDRAIL_DATABASE_URL',
            'sqlite:///guardrails.db'
        )
        
        # SQLite-specific settings
        connect_args = {}
        if self.connection_string.startswith('sqlite'):
            connect_args['check_same_thread'] = False
        
        self.engine = create_engine(
            self.connection_string,
            echo=echo,
            connect_args=connect_args,
            poolclass=StaticPool if 'sqlite' in self.connection_string else None
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
    
    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)."""
        Base.metadata.drop_all(bind=self.engine)
    
    @contextmanager
    def session(self):
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db.session() as session:
                session.query(...)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a new session (caller responsible for closing)."""
        return self.SessionLocal()


# Singleton database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """Get the singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
        _db_instance.create_tables()
    return _db_instance


def init_database(connection_string: Optional[str] = None, echo: bool = False) -> Database:
    """Initialize the database with custom settings."""
    global _db_instance
    _db_instance = Database(connection_string=connection_string, echo=echo)
    _db_instance.create_tables()
    return _db_instance
