from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import random
import time
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

torch.set_default_dtype(torch.float16)

# Check if CUDA is available and set device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')  # Use the first GPU
# Example: Creating a tensor on the default device (CUDA or CPU)
tensor = torch.zeros(2, 2, device=device)
# Initialize the embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models as well

model_name = 'Snowflake/snowflake-arctic-embed-m-v2.0'

model = SentenceTransformer(model_name, trust_remote_code=True)
model = model.to(device)
# Define log levels and their probability distribution
LOG_LEVELS = {
    "CRITICAL": 10,
    "WARNING": 20,
    "INFO": 70,
    "ERROR": 10
}

# Normalize probabilities
TOTAL = sum(LOG_LEVELS.values())
LOG_LEVEL_PROB = {level: weight / TOTAL for level, weight in LOG_LEVELS.items()}

# Predefined error messages for different logs
ERRORS = {
    "application_logs": {
        "CRITICAL": ["Database connection failed", "Application crashed", "Payment gateway down"],
        "WARNING": ["High response time", "Slow query detected", "Memory usage high"],
        "INFO": ["User logged in", "Page loaded successfully", "Cache hit rate optimal"]
    },
    "kubernetes_logs": {
        "CRITICAL": ["API server unresponsive", "Pod crash loop detected", "Node not reachable", "Etcd data corruption"],
        "WARNING": ["Resource quota exceeded", "Pod eviction", "Disk pressure detected", "High memory usage"],
        "INFO": ["Pod started successfully", "Node joined cluster", "Service discovery successful", "Replica set scaling"]
    },
    "apache_logs": {
        "CRITICAL": ["Apache server down", "Failed to load configuration", "Module failure"],
        "WARNING": ["Request time exceeded", "Suspicious request detected", "SSL certificate error"],
        "INFO": ["Request served", "Apache started successfully", "Configuration loaded"]
    },
    "httpd_logs": {
        "CRITICAL": ["Backend API crash", "Timeout on API request", "Database connection failed"],
        "WARNING": ["High API response time", "Failed API request", "API server memory usage high"],
        "INFO": ["API request received", "API request successful", "User request processed"]
    },
    "nginx_logs": {
        "CRITICAL": ["Nginx server down", "Failed to start Nginx", "SSL handshake failure"],
        "WARNING": ["Request timeout", "Service unavailable", "Upstream server not reachable"],
        "INFO": ["Request received", "Request served", "Nginx server restarted"]
    },
    "postgresql_logs": {
        "CRITICAL": ["Database connection failed", "Query timeout", "Database crash"],
        "WARNING": ["High query execution time", "Slow index usage", "Disk space low"],
        "INFO": ["Query executed successfully", "Database connection established", "Index created"]
    },
    "redis_logs": {
        "CRITICAL": ["Redis server down", "Memory limit reached", "Persistence error"],
        "WARNING": ["Low memory warning", "Slow log detected", "Eviction policy reached"],
        "INFO": ["Key set successfully", "Cache hit rate optimal", "Redis connection established"]
    },
    "kafka_logs": {
        "CRITICAL": ["Kafka broker unresponsive", "Message queue overflow", "Topic creation failed"],
        "WARNING": ["High message latency", "Low partition replication", "Consumer lag detected"],
        "INFO": ["Message sent to topic", "Consumer connected", "Kafka broker restarted"]
    }
}

# Application categories for different logs
APPLICATIONS = {
    "application_logs": {
        "frontend": ["ui-service", "web-client"],
        "backend": ["auth-service", "order-service", "payment-gateway"],
        "database": ["mysql-db", "redis-cache"],
        "messaging": ["kafka-broker", "rabbitmq-server"]
    },
    "kubernetes_logs": {
        "kubernetes": ["kube-apiserver", "kube-scheduler", "kube-controller-manager", "kube-proxy", "etcd"],
        "storage": ["persistent-volume", "ceph", "glusterfs"],
        "networking": ["calico", "flannel", "weave-net"],
        "monitoring": ["prometheus", "grafana", "alertmanager"]
    },
    "apache_logs": {
        "frontend": ["apache-frontend-server"],
        "backend": ["apache-backend-server"]
    },
    "httpd_logs": {
        "backend_api": ["httpd-api-server"]
    },
    "nginx_logs": {
        "load_balancer": ["nginx-lb-server"]
    },
    "postgresql_logs": {
        "database": ["postgres-db"]
    },
    "redis_logs": {
        "cache": ["redis-server"]
    },
    "kafka_logs": {
        "messaging_queue": ["kafka-broker"]
    }
}

def connect_to_milvus(host: str, port: str):
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")

def create_collection(collection_name: str):
    fields = [
        FieldSchema(name="log_id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="timestamp", dtype=DataType.INT64),
        FieldSchema(name="service", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="log_level", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="host", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="pod_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="trace_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="raw_log", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Example embedding size
    ]
    schema = CollectionSchema(fields, description=f"Logs for {collection_name}")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created successfully!")

def generate_log(collection_type: str):
    """Generates a single log entry."""
    log_level = random.choices(list(LOG_LEVEL_PROB.keys()), weights=LOG_LEVEL_PROB.values())[0]
    app_category = random.choice(list(APPLICATIONS[collection_type].keys()))
    application = random.choice(APPLICATIONS[collection_type][app_category])
    error_message = random.choice(ERRORS[collection_type][log_level])
    
    log_entry = {
        "timestamp": int(time.time() * 1000),  # Unix timestamp in milliseconds
        "level": log_level,
        "application": application,
        "category": app_category,
        "message": error_message 
    }
    return log_entry

def generate_log_embedding(log_message):
    """Generates embedding for the log message."""
    return model.encode(log_message).tolist()

def insert_log_to_milvus(log_entry, collection_name):
    """Inserts a log entry into Milvus."""
    embedding = generate_log_embedding(log_entry["message"])

    # Prepare the data with all required fields
    data = [
        [log_entry.get("log_id", random.randint(1, 1000000))],  # Generate a random log_id if not provided
        [log_entry["timestamp"]],
        [log_entry["application"]],
        [log_entry["level"]],
        [log_entry.get("host", "default_host")],  # Add default if not provided
        [log_entry.get("namespace", "default_namespace")],  # Add default if not provided
        [log_entry.get("pod_name", "default_pod")],  # Add default if not provided
        [log_entry.get("trace_id", str(random.randint(1, 1000000)))],  # Generate a random trace_id if not provided
        [log_entry["message"]],
        [embedding]
    ]
    
    collection = Collection(collection_name)
    collection.insert(data)
    print(f"Inserted log into Milvus: {log_entry['message']}")

def stream_logs(collection_type: str, collection_name: str):
    """Continuously generates and streams logs."""
    for _ in range(10):  # Loop 1000 times
        log_entry = generate_log(collection_type)  # Generate a log entry
        insert_log_to_milvus(log_entry, collection_name)  # Insert the log into Milvus collection
        time.sleep(random.uniform(0.5, 2))  # Simulate streaming interval
    print(f"Streamed 1000 {collection_type} logs.")

if __name__ == "__main__":
    MILVUS_HOST = "192.168.2.220"
    MILVUS_PORT = "19530"
    
    # Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Collections and their types
    collections = {
        "application_logs": "application_logs",
        "kubernetes_logs": "kubernetes_logs",
        "apache_logs": "apache_logs",
        "httpd_logs": "httpd_logs",
        "nginx_logs": "nginx_logs",
        "postgresql_logs": "postgresql_logs",
        "redis_logs": "redis_logs",
        "kafka_logs": "kafka_logs"
    }

    # Create collections if they don't exist
    for collection_name in collections.values():
        if collection_name not in utility.list_collections():
            create_collection(collection_name)
        else:
            print(f"Collection '{collection_name}' already exists.")
    
    # Start streaming logs for each collection
    for collection_name, collection_type in collections.items():
        stream_logs(collection_type, collection_name)
