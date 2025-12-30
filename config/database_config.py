import os
from typing import Optional, Dict, Any
from pathlib import Path

class DatabaseConfig:
    
    DEFAULT_CONFIG = {
        'host': 'localhost',
        'port': 5432,
        'database': 'llm_core',
        'user': 'postgres',
        'sslmode': 'prefer'
    }

    @classmethod
    def get_connection_string(cls) -> str:
        
        connection_string = os.getenv('POSTGRESQL_CONNECTION_STRING')
        if connection_string:
            return connection_string

        config = cls._get_config_from_env()

        return cls._config_to_connection_string(config)

    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        
        connection_string = os.getenv('POSTGRESQL_CONNECTION_STRING')
        if connection_string:
            config = cls._parse_connection_string(connection_string)
            if not config.get('password'):
                raise ValueError("Password not found in POSTGRESQL_CONNECTION_STRING")
            return config

        env_config = cls._load_from_env_file(cls.DEFAULT_CONFIG.copy())
        if env_config != cls.DEFAULT_CONFIG:
            if not env_config.get('password'):
                raise ValueError("POSTGRESQL_PASSWORD not found in .env file")
            return env_config

        config = cls._get_config_from_env()
        if not config.get('password'):
            raise ValueError("POSTGRESQL_PASSWORD environment variable not found")
        return config

    @classmethod
    def _get_config_from_env(cls) -> Dict[str, Any]:
        
        config = cls.DEFAULT_CONFIG.copy()

        env_mappings = {
            'POSTGRESQL_HOST': 'host',
            'POSTGRESQL_PORT': 'port',
            'POSTGRESQL_DATABASE': 'database',
            'POSTGRESQL_USER': 'user',
            'POSTGRESQL_PASSWORD': 'password',
            'POSTGRESQL_SSLMODE': 'sslmode'
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if config_key == 'port':
                    try:
                        config[config_key] = int(value)
                    except ValueError:
                        config[config_key] = value
                else:
                    config[config_key] = value

        return config

    @classmethod
    def _load_from_env_file(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        
        env_file = Path(__file__).parent.parent / '.env'

        if not env_file.exists():
            return config

        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        if key == 'POSTGRESQL_CONNECTION_STRING':
                            parsed = cls._parse_connection_string(value)
                            config.clear()
                            config.update(parsed)
                            return config
                        elif key.startswith('POSTGRESQL_'):
                            config_key = key.replace('POSTGRESQL_', '').lower()
                            if config_key in ['host', 'database', 'user', 'password', 'sslmode']:
                                config[config_key] = value
                            elif config_key == 'port':
                                try:
                                    config[config_key] = int(value)
                                except ValueError:
                                    config[config_key] = value

        except Exception as e:
            print(f"Warning: Could not load .env file: {e}")

        return config

    @classmethod
    def _config_to_connection_string(cls, config: Dict[str, Any]) -> str:
        
        parts = []

        if config.get('host'):
            parts.append(f"host={config['host']}")

        if config.get('port'):
            parts.append(f"port={config['port']}")

        if config.get('database'):
            parts.append(f"dbname={config['database']}")

        if config.get('user'):
            parts.append(f"user={config['user']}")

        if config.get('password'):
            parts.append(f"password={config['password']}")

        if config.get('sslmode'):
            parts.append(f"sslmode={config['sslmode']}")

        return ' '.join(parts)

    @classmethod
    def _parse_connection_string(cls, conn_str: str) -> Dict[str, Any]:
        
        config = cls.DEFAULT_CONFIG.copy()

        pairs = conn_str.split()
        for pair in pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
                if key == 'port':
                    try:
                        config[key] = int(value)
                    except ValueError:
                        config[key] = value
                else:
                    config[key] = value

        return config

    @classmethod
    def get_safe_connection_string(cls, mask_password: bool = True) -> str:
        
        config = cls.get_config_dict()

        if mask_password and 'password' in config:
            config_copy = config.copy()
            config_copy['password'] = '***MASKED***'
            return cls._config_to_connection_string(config_copy)

        return cls._config_to_connection_string(config)

    @classmethod
    def validate_connection(cls) -> bool:
        
        try:
            import psycopg2
            conn_str = cls.get_connection_string()
            conn = psycopg2.connect(conn_str)
            conn.close()
            return True
        except ValueError as e:
            print(f"Database configuration error: {e}")
            return False
        except Exception as e:
            print(f"Database connection validation failed: {e}")
            return False

    @classmethod
    def print_config(cls):
        
        safe_conn = cls.get_safe_connection_string()
        config = cls.get_config_dict()

        print("🔧 Database Configuration:")
        print(f"  Connection String: {safe_conn}")
        print(f"  Host: {config.get('host', 'N/A')}")
        print(f"  Port: {config.get('port', 'N/A')}")
        print(f"  Database: {config.get('database', 'N/A')}")
        print(f"  User: {config.get('user', 'N/A')}")
        print(f"  SSL Mode: {config.get('sslmode', 'N/A')}")

        if os.getenv('POSTGRESQL_CONNECTION_STRING'):
            print("  Source: POSTGRESQL_CONNECTION_STRING environment variable")
        elif (Path(__file__).parent.parent / '.env').exists():
            print("  Source: .env file (recommended)")
        elif any(os.getenv(f'POSTGRESQL_{k.upper()}') for k in ['host', 'port', 'database', 'user', 'password']):
            print("  Source: Individual POSTGRESQL_* environment variables")
        else:
            print("  ⚠️  Source: Default configuration (not recommended for production)")
            print("  💡 Create a .env file with your database credentials")

def get_postgresql_connection_string() -> str:
    return DatabaseConfig.get_connection_string()

def get_postgresql_config() -> Dict[str, Any]:
    return DatabaseConfig.get_config_dict()

def validate_postgresql_connection() -> bool:
    return DatabaseConfig.validate_connection()

def print_postgresql_config():
    DatabaseConfig.print_config()

if __name__ == "__main__":
    print("Testing Database Configuration...")
    print_postgresql_config()

    print(f"\nConnection validation: {'✅ Success' if validate_postgresql_connection() else '❌ Failed'}")