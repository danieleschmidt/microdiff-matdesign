"""Database connection management."""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import json

import numpy as np


logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_path: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_path: Path to SQLite database file
        """
        if database_path is None:
            database_path = os.getenv('MICRODIFF_DB_PATH', 'microdiff_data.db')
        
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database schema
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database schema."""
        
        schema_sql = """
        -- Experiments table
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            alloy VARCHAR(100) NOT NULL,
            process VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        );
        
        -- Microstructures table
        CREATE TABLE IF NOT EXISTS microstructures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            name VARCHAR(255) NOT NULL,
            volume_data_path VARCHAR(500),
            voxel_size REAL,
            dimensions JSON,
            acquisition_method VARCHAR(100),
            preprocessing_applied JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        );
        
        -- Process parameters table
        CREATE TABLE IF NOT EXISTS process_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            microstructure_id INTEGER,
            laser_power REAL,
            scan_speed REAL,
            layer_thickness REAL,
            hatch_spacing REAL,
            powder_bed_temp REAL,
            atmosphere VARCHAR(50),
            scan_strategy VARCHAR(100),
            additional_params JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id),
            FOREIGN KEY (microstructure_id) REFERENCES microstructures (id)
        );
        
        -- Material properties table
        CREATE TABLE IF NOT EXISTS material_properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            microstructure_id INTEGER,
            tensile_strength REAL,
            yield_strength REAL,
            elongation REAL,
            hardness REAL,
            density REAL,
            elastic_modulus REAL,
            thermal_conductivity REAL,
            test_method VARCHAR(100),
            test_conditions JSON,
            measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id),
            FOREIGN KEY (microstructure_id) REFERENCES microstructures (id)
        );
        
        -- Analysis results table
        CREATE TABLE IF NOT EXISTS analysis_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            microstructure_id INTEGER NOT NULL,
            analysis_type VARCHAR(100) NOT NULL,
            features JSON NOT NULL,
            quality_metrics JSON,
            recommendations JSON,
            warnings JSON,
            confidence_scores JSON,
            analysis_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (microstructure_id) REFERENCES microstructures (id)
        );
        
        -- Model predictions table
        CREATE TABLE IF NOT EXISTS model_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50),
            input_data JSON,
            predicted_parameters JSON,
            uncertainty_metrics JSON,
            confidence_score REAL,
            prediction_time REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id)
        );
        
        -- Training data table
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            microstructure_id INTEGER NOT NULL,
            parameter_id INTEGER NOT NULL,
            is_validation BOOLEAN DEFAULT FALSE,
            is_test BOOLEAN DEFAULT FALSE,
            quality_score REAL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (microstructure_id) REFERENCES microstructures (id),
            FOREIGN KEY (parameter_id) REFERENCES process_parameters (id)
        );
        
        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_experiments_alloy_process ON experiments (alloy, process);
        CREATE INDEX IF NOT EXISTS idx_microstructures_experiment ON microstructures (experiment_id);
        CREATE INDEX IF NOT EXISTS idx_parameters_experiment ON process_parameters (experiment_id);
        CREATE INDEX IF NOT EXISTS idx_parameters_microstructure ON process_parameters (microstructure_id);
        CREATE INDEX IF NOT EXISTS idx_properties_experiment ON material_properties (experiment_id);
        CREATE INDEX IF NOT EXISTS idx_properties_microstructure ON material_properties (microstructure_id);
        CREATE INDEX IF NOT EXISTS idx_analysis_microstructure ON analysis_results (microstructure_id);
        CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_results (analysis_type);
        CREATE INDEX IF NOT EXISTS idx_predictions_experiment ON model_predictions (experiment_id);
        CREATE INDEX IF NOT EXISTS idx_predictions_model ON model_predictions (model_name, model_version);
        CREATE INDEX IF NOT EXISTS idx_training_data ON training_data (microstructure_id, parameter_id);
        """
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Execute schema creation in chunks
            for statement in schema_sql.split(';'):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)
            conn.commit()
            
        logger.info(f"Database initialized at {self.database_path}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection context manager."""
        
        conn = sqlite3.connect(
            self.database_path,
            timeout=30.0,
            isolation_level=None  # Autocommit mode
        )
        
        try:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            # Enable WAL mode for better concurrency
            conn.execute("PRAGMA journal_mode = WAL")
            # Row factory for dict-like access
            conn.row_factory = sqlite3.Row
            
            yield conn
            
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> list:
        """Execute a SELECT query and return results."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an UPDATE/INSERT/DELETE query and return affected rows."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.rowcount
    
    def execute_insert(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT query and return the last row ID."""
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.lastrowid
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database.
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            True if backup successful, False otherwise
        """
        
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self.get_connection() as source:
                with sqlite3.connect(backup_path) as backup:
                    source.backup(backup)
            
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def restore_database(self, backup_path: str) -> bool:
        """Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful, False otherwise
        """
        
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Remove current database
            if self.database_path.exists():
                self.database_path.unlink()
            
            # Copy backup to current location
            import shutil
            shutil.copy2(backup_path, self.database_path)
            
            logger.info(f"Database restored from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        
        stats = {}
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Table counts
                tables = [
                    'experiments', 'microstructures', 'process_parameters',
                    'material_properties', 'analysis_results', 'model_predictions',
                    'training_data'
                ]
                
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    stats[f'{table}_count'] = count
                
                # Database size
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                stats['database_size_bytes'] = page_size * page_count
                stats['database_size_mb'] = (page_size * page_count) / (1024 * 1024)
                
                # Vacuum stats
                cursor.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]
                stats['free_pages'] = freelist_count
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def vacuum_database(self) -> bool:
        """Vacuum the database to reclaim space."""
        
        try:
            with self.get_connection() as conn:
                conn.execute("VACUUM")
            
            logger.info("Database vacuumed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")
            return False
    
    def migrate_database(self, target_version: str) -> bool:
        """Migrate database schema to target version.
        
        Args:
            target_version: Target schema version
            
        Returns:
            True if migration successful, False otherwise
        """
        
        # This is a placeholder for future schema migrations
        # In a real implementation, you would have version tracking
        # and migration scripts
        
        logger.info(f"Database migration to version {target_version} not implemented")
        return True


class JSONAdapter:
    """JSON adapter for SQLite storage."""
    
    @staticmethod
    def adapt_json(obj) -> str:
        """Adapt Python object to JSON string."""
        if isinstance(obj, np.ndarray):
            return json.dumps(obj.tolist())
        return json.dumps(obj)
    
    @staticmethod
    def convert_json(s: bytes) -> Any:
        """Convert JSON string to Python object."""
        return json.loads(s.decode('utf-8'))


# Register JSON adapters
sqlite3.register_adapter(dict, JSONAdapter.adapt_json)
sqlite3.register_adapter(list, JSONAdapter.adapt_json)
sqlite3.register_converter("JSON", JSONAdapter.convert_json)