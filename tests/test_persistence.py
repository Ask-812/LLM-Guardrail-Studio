"""
Unit tests for persistence layer
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
from datetime import datetime

from persistence.database import Database, EvaluationRecord, RuleRecord, ConfigRecord, AuditLog
from persistence.repositories import (
    EvaluationRepository,
    RuleRepository, 
    ConfigRepository,
    AuditRepository
)


class TestDatabase(unittest.TestCase):
    """Test cases for Database class"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(f"sqlite:///{self.db_path}")
    
    def tearDown(self):
        """Clean up test database"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_database_creation(self):
        """Test database file is created"""
        self.assertTrue(os.path.exists(self.db_path))
    
    def test_session_management(self):
        """Test session creation and management"""
        with self.db.session() as session:
            self.assertIsNotNone(session)
    
    def test_tables_created(self):
        """Test that all tables are created"""
        with self.db.session() as session:
            # Tables should exist
            from sqlalchemy import inspect
            inspector = inspect(session.get_bind())
            tables = inspector.get_table_names()
            
            self.assertIn("evaluation_records", tables)
            self.assertIn("rule_records", tables)
            self.assertIn("config_records", tables)
            self.assertIn("audit_logs", tables)


class TestEvaluationRepository(unittest.TestCase):
    """Test cases for EvaluationRepository"""
    
    def setUp(self):
        """Set up test database and repository"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(f"sqlite:///{self.db_path}")
        self.repo = EvaluationRepository(self.db)
    
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_create_evaluation(self):
        """Test creating an evaluation record"""
        record = self.repo.create(
            prompt_hash="hash123",
            response_hash="hash456",
            passed=True,
            scores={"toxicity": 0.1, "alignment": 0.8},
            flags=[]
        )
        
        self.assertIsNotNone(record.id)
        self.assertEqual(record.prompt_hash, "hash123")
        self.assertTrue(record.passed)
    
    def test_get_evaluation_by_id(self):
        """Test retrieving evaluation by ID"""
        created = self.repo.create(
            prompt_hash="hash1",
            response_hash="hash2",
            passed=True,
            scores={},
            flags=[]
        )
        
        retrieved = self.repo.get_by_id(created.id)
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, created.id)
    
    def test_list_evaluations(self):
        """Test listing evaluations with pagination"""
        # Create multiple records
        for i in range(5):
            self.repo.create(
                prompt_hash=f"hash{i}",
                response_hash=f"resp{i}",
                passed=i % 2 == 0,
                scores={"toxicity": i * 0.1},
                flags=[]
            )
        
        # List all
        all_records = self.repo.list(limit=10)
        self.assertEqual(len(all_records), 5)
        
        # List with limit
        limited = self.repo.list(limit=3)
        self.assertEqual(len(limited), 3)
    
    def test_filter_by_passed(self):
        """Test filtering evaluations by passed status"""
        for i in range(4):
            self.repo.create(
                prompt_hash=f"hash{i}",
                response_hash=f"resp{i}",
                passed=i < 2,
                scores={},
                flags=[]
            )
        
        passed = self.repo.list(passed_only=True)
        failed = self.repo.list(failed_only=True)
        
        self.assertEqual(len(passed), 2)
        self.assertEqual(len(failed), 2)
    
    def test_get_statistics(self):
        """Test getting evaluation statistics"""
        for i in range(10):
            self.repo.create(
                prompt_hash=f"hash{i}",
                response_hash=f"resp{i}",
                passed=i < 7,
                scores={"toxicity": i * 0.1},
                flags=[] if i < 5 else ["flag"]
            )
        
        stats = self.repo.get_statistics()
        
        self.assertEqual(stats["total"], 10)
        self.assertEqual(stats["passed"], 7)
        self.assertEqual(stats["failed"], 3)
        self.assertAlmostEqual(stats["pass_rate"], 0.7, places=2)


class TestRuleRepository(unittest.TestCase):
    """Test cases for RuleRepository"""
    
    def setUp(self):
        """Set up test database and repository"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(f"sqlite:///{self.db_path}")
        self.repo = RuleRepository(self.db)
    
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_create_rule(self):
        """Test creating a rule"""
        rule = self.repo.create(
            name="no_spam",
            rule_type="keyword",
            pattern="spam",
            action="flag",
            description="No spam allowed"
        )
        
        self.assertIsNotNone(rule.id)
        self.assertEqual(rule.name, "no_spam")
        self.assertTrue(rule.enabled)
    
    def test_get_rule_by_name(self):
        """Test retrieving rule by name"""
        self.repo.create(
            name="test_rule",
            rule_type="regex",
            pattern=r"\d+",
            action="flag"
        )
        
        retrieved = self.repo.get_by_name("test_rule")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test_rule")
    
    def test_update_rule(self):
        """Test updating a rule"""
        rule = self.repo.create(
            name="update_me",
            rule_type="keyword",
            pattern="old",
            action="flag"
        )
        
        updated = self.repo.update(
            rule.id,
            pattern="new",
            action="block"
        )
        
        self.assertEqual(updated.pattern, "new")
        self.assertEqual(updated.action, "block")
    
    def test_delete_rule(self):
        """Test deleting a rule"""
        rule = self.repo.create(
            name="delete_me",
            rule_type="keyword",
            pattern="test",
            action="flag"
        )
        
        result = self.repo.delete(rule.id)
        self.assertTrue(result)
        
        retrieved = self.repo.get_by_name("delete_me")
        self.assertIsNone(retrieved)
    
    def test_list_enabled_rules(self):
        """Test listing only enabled rules"""
        self.repo.create(name="enabled1", rule_type="keyword", pattern="a", action="flag")
        self.repo.create(name="enabled2", rule_type="keyword", pattern="b", action="flag")
        disabled = self.repo.create(name="disabled", rule_type="keyword", pattern="c", action="flag")
        
        self.repo.update(disabled.id, enabled=False)
        
        enabled_rules = self.repo.list(enabled_only=True)
        self.assertEqual(len(enabled_rules), 2)


class TestConfigRepository(unittest.TestCase):
    """Test cases for ConfigRepository"""
    
    def setUp(self):
        """Set up test database and repository"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(f"sqlite:///{self.db_path}")
        self.repo = ConfigRepository(self.db)
    
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_set_config(self):
        """Test setting a config value"""
        self.repo.set("toxicity_threshold", "0.7")
        
        value = self.repo.get("toxicity_threshold")
        self.assertEqual(value, "0.7")
    
    def test_update_config(self):
        """Test updating a config value"""
        self.repo.set("key", "value1")
        self.repo.set("key", "value2")
        
        value = self.repo.get("key")
        self.assertEqual(value, "value2")
    
    def test_get_nonexistent_config(self):
        """Test getting non-existent config returns None"""
        value = self.repo.get("nonexistent")
        self.assertIsNone(value)
    
    def test_get_with_default(self):
        """Test getting config with default value"""
        value = self.repo.get("nonexistent", default="default_value")
        self.assertEqual(value, "default_value")
    
    def test_list_all_config(self):
        """Test listing all config values"""
        self.repo.set("key1", "val1")
        self.repo.set("key2", "val2")
        self.repo.set("key3", "val3")
        
        all_config = self.repo.list()
        
        self.assertEqual(len(all_config), 3)
        self.assertIn("key1", all_config)
        self.assertEqual(all_config["key1"], "val1")
    
    def test_delete_config(self):
        """Test deleting a config value"""
        self.repo.set("to_delete", "value")
        result = self.repo.delete("to_delete")
        
        self.assertTrue(result)
        self.assertIsNone(self.repo.get("to_delete"))


class TestAuditRepository(unittest.TestCase):
    """Test cases for AuditRepository"""
    
    def setUp(self):
        """Set up test database and repository"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(f"sqlite:///{self.db_path}")
        self.repo = AuditRepository(self.db)
    
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_log_action(self):
        """Test logging an action"""
        log = self.repo.log(
            action="evaluation",
            details={"prompt_hash": "hash123", "passed": True}
        )
        
        self.assertIsNotNone(log.id)
        self.assertEqual(log.action, "evaluation")
    
    def test_log_with_user(self):
        """Test logging with user context"""
        log = self.repo.log(
            action="config_change",
            details={"key": "threshold", "old": "0.5", "new": "0.7"},
            user="admin"
        )
        
        self.assertEqual(log.user, "admin")
    
    def test_list_logs_by_action(self):
        """Test filtering logs by action"""
        self.repo.log(action="evaluation", details={})
        self.repo.log(action="evaluation", details={})
        self.repo.log(action="config_change", details={})
        
        eval_logs = self.repo.list(action="evaluation")
        
        self.assertEqual(len(eval_logs), 2)
    
    def test_list_logs_with_limit(self):
        """Test listing logs with limit"""
        for i in range(10):
            self.repo.log(action="test", details={"i": i})
        
        logs = self.repo.list(limit=5)
        
        self.assertEqual(len(logs), 5)
    
    def test_logs_ordered_by_timestamp(self):
        """Test logs are ordered by timestamp descending"""
        self.repo.log(action="first", details={})
        self.repo.log(action="second", details={})
        self.repo.log(action="third", details={})
        
        logs = self.repo.list()
        
        # Most recent should be first
        self.assertEqual(logs[0].action, "third")


class TestDatabaseIntegration(unittest.TestCase):
    """Integration tests for database operations"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.db = Database(f"sqlite:///{self.db_path}")
        
        self.eval_repo = EvaluationRepository(self.db)
        self.rule_repo = RuleRepository(self.db)
        self.config_repo = ConfigRepository(self.db)
        self.audit_repo = AuditRepository(self.db)
    
    def tearDown(self):
        """Clean up"""
        self.db.close()
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)
        os.rmdir(self.temp_dir)
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow"""
        # Set up config
        self.config_repo.set("toxicity_threshold", "0.7")
        
        # Create a rule
        rule = self.rule_repo.create(
            name="no_spam",
            rule_type="keyword",
            pattern="spam",
            action="flag"
        )
        
        # Create evaluation
        evaluation = self.eval_repo.create(
            prompt_hash="prompt123",
            response_hash="response456",
            passed=True,
            scores={"toxicity": 0.1, "alignment": 0.9},
            flags=[]
        )
        
        # Log the action
        self.audit_repo.log(
            action="evaluation",
            details={
                "evaluation_id": evaluation.id,
                "rules_applied": [rule.name]
            }
        )
        
        # Verify
        stats = self.eval_repo.get_statistics()
        self.assertEqual(stats["total"], 1)
        
        logs = self.audit_repo.list(action="evaluation")
        self.assertEqual(len(logs), 1)


if __name__ == "__main__":
    unittest.main()
