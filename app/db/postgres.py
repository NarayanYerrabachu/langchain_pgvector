"""PostgreSQL Vector Database Connection."""
import psycopg2
from psycopg2.extras import execute_values
from psycopg2 import pool
import logging
from typing import List, Dict, Any, Tuple
import json

logger = logging.getLogger(__name__)


class PostgresVdbClient:
    """PostgreSQL Vector Database Client with pgvector support."""

    def __init__(
            self,
            engine_name: str,
            dbname: str = "rag_db",
            user: str = "postgres",
            password: str = "postgres",
            host: str = "localhost",
            port: int = 5432,
            min_conn: int = 1,
            max_conn: int = 10
    ):
        """
        Initialize PostgreSQL connection pool.

        Args:
            engine_name: Name of the engine (for logging)
            dbname: Database name
            user: Database user
            password: Database password
            host: Database host
            port: Database port
            min_conn: Minimum connections in pool
            max_conn: Maximum connections in pool
        """
        self.engine_name = engine_name

        try:
            # Create connection pool
            self.pool = psycopg2.pool.SimpleConnectionPool(
                min_conn,
                max_conn,
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
            )
            logger.info(f"✅ PostgreSQL connection pool created for {engine_name}")

            # Enable pgvector extension
            self._enable_pgvector()

        except Exception as e:
            logger.error(f"❌ Failed to create connection pool: {e}", exc_info=True)
            raise

    def _get_connection(self):
        """Get connection from pool."""
        return self.pool.getconn()

    def _return_connection(self, conn):
        """Return connection to pool."""
        self.pool.putconn(conn)

    def _enable_pgvector(self):
        """Enable pgvector extension."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()

            logger.info("✅ pgvector extension enabled")

            cursor.close()
            self._return_connection(conn)
        except Exception as e:
            logger.warning(f"⚠️ Could not enable pgvector: {e}")
            self._return_connection(conn)

    def create_table(self, table_name: str, vector_dim: int = 1536):
        """
        Create vector table if it doesn't exist.

        Args:
            table_name: Name of the table
            vector_dim: Dimension of embedding vectors (default: 1536 for OpenAI)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            sql = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding vector({vector_dim}),
                metadata JSONB DEFAULT '{{}}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx 
            ON {table_name} 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """

            cursor.execute(sql)
            conn.commit()

            logger.info(f"✅ Table '{table_name}' created/verified with vector dimension {vector_dim}")

            cursor.close()
            self._return_connection(conn)
        except Exception as e:
            logger.error(f"❌ Error creating table '{table_name}': {e}", exc_info=True)
            self._return_connection(conn)
            raise

    def batch_exists_check(self, table_name: str, ids: List[str]) -> Dict[str, bool]:
        """
        Check which IDs already exist in table.

        Args:
            table_name: Name of the table
            ids: List of IDs to check

        Returns:
            Dictionary mapping ID to existence boolean
        """
        if not ids:
            return {}

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create placeholders
            placeholders = ','.join(['%s'] * len(ids))
            sql = f"SELECT id FROM {table_name} WHERE id IN ({placeholders});"

            cursor.execute(sql, ids)
            existing_ids = {row[0] for row in cursor.fetchall()}

            result = {id: id in existing_ids for id in ids}

            cursor.close()
            self._return_connection(conn)

            return result
        except Exception as e:
            logger.error(f"❌ Error checking IDs: {e}", exc_info=True)
            self._return_connection(conn)
            return {id: False for id in ids}

    def batch_insert(
            self,
            table_name: str,
            columns: List[str],
            data: List[Tuple]
    ) -> int:
        """
        Batch insert data into table.

        Args:
            table_name: Name of the table
            columns: List of column names
            data: List of tuples with data

        Returns:
            Number of rows inserted
        """
        if not data:
            return 0

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Build INSERT statement
            col_names = ', '.join(columns)
            placeholders = ', '.join(['%s'] * len(columns))
            sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders}) ON CONFLICT (id) DO NOTHING;"

            # Execute batch insert
            execute_values(cursor, sql, data, page_size=1000)
            conn.commit()

            rows_inserted = cursor.rowcount
            logger.info(f"✅ Inserted {rows_inserted} rows into '{table_name}'")

            cursor.close()
            self._return_connection(conn)

            return rows_inserted
        except Exception as e:
            logger.error(f"❌ Error inserting data: {e}", exc_info=True)
            self._return_connection(conn)
            raise

    def search(
            self,
            table_name: str,
            query_embedding: List[float],
            limit: int = 5,
            threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using pgvector.

        Args:
            table_name: Name of the table
            query_embedding: Query embedding vector
            limit: Number of results to return
            threshold: Minimum similarity threshold

        Returns:
            List of matching documents with similarity scores
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Convert embedding to pgvector format
            embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

            sql = f"""
            SELECT id, text, metadata, 
                   1 - (embedding <=> %s::vector) as similarity
            FROM {table_name}
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """

            cursor.execute(sql, (embedding_str, embedding_str, threshold, embedding_str, limit))
            results = cursor.fetchall()

            # Format results
            formatted_results = [
                {
                    "id": row[0],
                    "text": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "similarity": float(row[3])
                }
                for row in results
            ]

            logger.debug(f"✅ Search found {len(formatted_results)} results")

            cursor.close()
            self._return_connection(conn)

            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error searching: {e}", exc_info=True)
            self._return_connection(conn)
            return []

    def get_all(self, table_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all documents from table.

        Args:
            table_name: Name of the table
            limit: Maximum number of documents to return

        Returns:
            List of documents
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            sql = f"""
            SELECT id, text, metadata, created_at
            FROM {table_name}
            ORDER BY created_at DESC
            LIMIT %s;
            """

            cursor.execute(sql, (limit,))
            results = cursor.fetchall()

            formatted_results = [
                {
                    "id": row[0],
                    "text": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "created_at": str(row[3])
                }
                for row in results
            ]

            cursor.close()
            self._return_connection(conn)

            return formatted_results
        except Exception as e:
            logger.error(f"❌ Error fetching all: {e}", exc_info=True)
            self._return_connection(conn)
            return []

    def delete_by_id(self, table_name: str, id: str) -> bool:
        """Delete document by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            sql = f"DELETE FROM {table_name} WHERE id = %s;"
            cursor.execute(sql, (id,))
            conn.commit()

            deleted = cursor.rowcount > 0
            logger.info(f"✅ Deleted document: {id}")

            cursor.close()
            self._return_connection(conn)

            return deleted
        except Exception as e:
            logger.error(f"❌ Error deleting: {e}", exc_info=True)
            self._return_connection(conn)
            return False

    def close(self):
        """Close all connections in pool."""
        try:
            self.pool.closeall()
            logger.info(f"✅ Connection pool closed for {self.engine_name}")
        except Exception as e:
            logger.error(f"❌ Error closing pool: {e}", exc_info=True)