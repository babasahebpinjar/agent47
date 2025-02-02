from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import random
import time
import json
import uuid
from sentence_transformers import SentenceTransformer

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Embedding model

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
import uuid

ERRORS = {
    "application_logs": {
        "CRITICAL": [
            "Database connection failed due to network timeout; retries exhausted after 5 attempts.",
            "Application crashed unexpectedly with segmentation fault in module 'libcore.so'; stack trace available.",
            "Payment gateway down: SSL handshake failed with error code 0x80004005; retrying in 5 minutes."
        ],
        "WARNING": [
            "High response time detected: Average API response time exceeded 5000ms for the last 10 minutes.",
            "Slow query detected: Query SELECT * FROM users WHERE id = ? took 12000ms to execute.",
            "Memory usage high: Application memory usage is at 85% of the allocated limit; consider scaling up."
        ],
        "INFO": [
            "User logged in successfully: User 'admin' authenticated via OAuth2; session ID generated: 12345.",
            "Page loaded successfully: Homepage rendered in 120ms with 15 database queries and 3 API calls.",
            "Cache hit rate optimal: Redis cache hit rate is at 95% for the last hour; performance is stable."
        ],
        "ERROR": [
            "Unexpected error occurred: Null pointer exception in function 'processRequest' at line 45; stack trace logged.",
            "Failed to process request: Request ID 67890 failed due to invalid JSON payload; returning 400 Bad Request.",
            "Internal server error: Database transaction rollback failed due to deadlock; manual intervention required."
        ]
    },
    "kubernetes_logs": {
        "CRITICAL": [
            "API server unresponsive: Kubernetes API server is not responding to health checks; restarting pod.",
            "Pod crash loop detected: Pod 'frontend-123' has crashed 5 times in the last 10 minutes.",
            "Node not reachable: Node 'worker-1' is unreachable due to network partition; investigating connectivity issues."
        ],
        "WARNING": [
            "Resource quota exceeded: Namespace 'default' has exceeded CPU quota; scaling down deployments.",
            "Pod eviction: Pod 'backend-456' evicted due to high memory usage on node 'worker-2'.",
            "Disk pressure detected: Node 'worker-3' is under disk pressure; freeing up space by deleting unused images."
        ],
        "INFO": [
            "Pod started successfully: Pod 'frontend-789' started in namespace 'production'; ready to serve traffic.",
            "Node joined cluster: Node 'worker-4' successfully joined the cluster; available for scheduling workloads.",
            "Service discovery successful: Service 'auth-service' discovered with 3 healthy endpoints; traffic routed accordingly."
        ],
        "ERROR": [
            "Kubernetes API error: Failed to create deployment 'backend' due to invalid YAML configuration.",
            "Pod creation failed: Pod 'frontend-123' failed to start due to image pull error; retrying.",
            "Cluster communication error: Etcd cluster is unreachable; API server cannot persist state changes."
        ]
    },
    "apache_logs": {
        "CRITICAL": [
            "Apache server down: Apache HTTP server crashed due to segmentation fault in module 'mod_ssl.so'.",
            "Failed to load configuration: Apache failed to load configuration file 'httpd.conf'; syntax error at line 45.",
            "Module failure: Module 'mod_rewrite' failed to initialize; Apache server cannot start without it."
        ],
        "WARNING": [
            "Request time exceeded: Request to '/api/v1/users' took 12000ms to complete; consider optimizing the endpoint.",
            "Suspicious request detected: Request from IP 192.168.1.100 contains potential SQL injection payload; blocked.",
            "SSL certificate error: SSL certificate for domain 'example.com' is expired; renew immediately to avoid downtime."
        ],
        "INFO": [
            "Request served: GET request to '/index.html' served in 50ms with status code 200.",
            "Apache started successfully: Apache HTTP server started on port 80; ready to accept connections.",
            "Configuration loaded: Apache configuration file 'httpd.conf' loaded successfully; server running with 10 worker threads."
        ],
        "ERROR": [
            "Apache configuration error: Directive 'DocumentRoot' in 'httpd.conf' points to non-existent directory '/var/www/html'.",
            "Failed to start Apache: Apache failed to bind to port 80; address already in use.",
            "Unexpected server error: Internal server error occurred while processing request to '/api/v1/data'; stack trace logged."
        ]
    },
    "httpd_logs": {
        "CRITICAL": [
            "Backend API crash: Backend API server crashed due to unhandled exception in function 'processRequest'.",
            "Timeout on API request: Request to '/api/v1/orders' timed out after 10000ms; backend unresponsive.",
            "Database connection failed: Failed to connect to MySQL database; connection pool exhausted."
        ],
        "WARNING": [
            "High API response time: Average response time for '/api/v1/users' exceeded 5000ms; investigate performance issues.",
            "Failed API request: POST request to '/api/v1/orders' failed with status code 500; retrying.",
            "API server memory usage high: Memory usage is at 90% of the allocated limit; consider scaling up."
        ],
        "INFO": [
            "API request received: GET request to '/api/v1/products' received; processing with query parameters.",
            "API request successful: POST request to '/api/v1/orders' completed with status code 201; order created.",
            "User request processed: User 'john_doe' successfully updated profile information; changes saved to database."
        ],
        "ERROR": [
            "HTTPD internal error: Internal server error occurred while processing request to '/api/v1/data'; stack trace logged.",
            "Failed to process API request: Request to '/api/v1/orders' failed due to invalid payload; returning 400 Bad Request.",
            "Unexpected server error: Database transaction failed due to deadlock; manual intervention required."
        ]
    },
    "nginx_logs": {
        "CRITICAL": [
            "Nginx server down: Nginx server crashed due to segmentation fault in module 'ngx_http_ssl_module'.",
            "Failed to start Nginx: Nginx failed to start due to invalid configuration in 'nginx.conf'.",
            "SSL handshake failure: SSL handshake failed for client 192.168.1.100; invalid certificate presented."
        ],
        "WARNING": [
            "Request timeout: Request to '/api/v1/data' timed out after 10000ms; upstream server unresponsive.",
            "Service unavailable: Upstream server 'backend-api' is down; returning 503 Service Unavailable to client.",
            "Upstream server not reachable: Upstream server 'auth-service' is unreachable; retrying in 5 seconds."
        ],
        "INFO": [
            "Request received: GET request to '/index.html' received from client 192.168.1.100; processing.",
            "Request served: GET request to '/index.html' served in 30ms with status code 200.",
            "Nginx server restarted: Nginx server restarted successfully; configuration changes applied."
        ],
        "ERROR": [
            "Nginx configuration error: Directive 'proxy_pass' in 'nginx.conf' points to invalid upstream server 'backend-api'.",
            "Failed to reload Nginx: Nginx failed to reload configuration due to syntax error in 'nginx.conf'.",
            "Unexpected server error: Internal server error occurred while processing request to '/api/v1/data'; stack trace logged."
        ]
    },
    "postgresql_logs": {
        "CRITICAL": [
            "Database connection failed: Failed to connect to PostgreSQL database; connection pool exhausted.",
            "Query timeout: Query 'SELECT * FROM users' timed out after 30000ms; consider optimizing the query.",
            "Database crash: PostgreSQL database crashed due to disk I/O error; restarting service."
        ],
        "WARNING": [
            "High query execution time: Query 'SELECT * FROM orders' took 15000ms to execute; consider adding indexes.",
            "Slow index usage: Index 'idx_users_email' is not being used in query 'SELECT * FROM users WHERE email = ?'.",
            "Disk space low: Disk space on '/var/lib/postgresql' is at 90% capacity; consider freeing up space."
        ],
        "INFO": [
            "Query executed successfully: Query 'INSERT INTO users (name, email) VALUES (?, ?)' completed in 50ms.",
            "Database connection established: Successfully connected to PostgreSQL database 'production' with user 'admin'.",
            "Index created: Index 'idx_orders_user_id' created on table 'orders'; query performance improved."
        ],
        "ERROR": [
            "PostgreSQL internal error: Internal server error occurred while executing query 'SELECT * FROM users'; stack trace logged.",
            "Failed to execute query: Query 'UPDATE users SET name = ? WHERE id = ?' failed due to deadlock.",
            "Unexpected database error: Database transaction failed due to constraint violation; manual intervention required."
        ]
    },
    "redis_logs": {
        "CRITICAL": [
            "Redis server down: Redis server crashed due to out-of-memory error; restarting service.",
            "Memory limit reached: Redis memory usage is at 100% of the allocated limit; evicting keys.",
            "Persistence error: Failed to persist data to disk; AOF rewrite failed due to disk I/O error."
        ],
        "WARNING": [
            "Low memory warning: Redis memory usage is at 85% of the allocated limit; consider scaling up.",
            "Slow log detected: Command 'KEYS *' took 5000ms to execute; consider optimizing the query.",
            "Eviction policy reached: Redis evicted 100 keys due to memory pressure; consider increasing memory limit."
        ],
        "INFO": [
            "Key set successfully: Key 'user:123' set with value 'john_doe'; TTL set to 3600 seconds.",
            "Cache hit rate optimal: Redis cache hit rate is at 95% for the last hour; performance is stable.",
            "Redis connection established: Successfully connected to Redis server 'redis-prod' with user 'admin'."
        ],
        "ERROR": [
            "Redis internal error: Internal server error occurred while executing command 'GET user:123'; stack trace logged.",
            "Failed to set key: Failed to set key 'user:123' due to memory allocation error; retrying.",
            "Unexpected cache error: Cache operation failed due to network partition; manual intervention required."
        ]
    },
    "kafka_logs": {
        "CRITICAL": [
            "Kafka broker unresponsive: Kafka broker 'broker-1' is not responding to health checks; restarting service.",
            "Message queue overflow: Kafka topic 'orders' has reached maximum queue size; messages are being dropped.",
            "Topic creation failed: Failed to create topic 'events' due to insufficient disk space on broker."
        ],
        "WARNING": [
            "High message latency: Average message latency for topic 'orders' exceeded 5000ms; investigate performance issues.",
            "Low partition replication: Partition 0 of topic 'events' has only 1 replica; consider increasing replication factor.",
            "Consumer lag detected: Consumer group 'order-processor' is lagging by 1000 messages; consider scaling up consumers."
        ],
        "INFO": [
            "Message sent to topic: Message with key 'order-123' successfully published to topic 'orders'; offset 456.",
            "Consumer connected: Consumer 'order-processor-1' successfully connected to Kafka cluster; assigned partitions [0, 1].",
            "Kafka broker restarted: Kafka broker 'broker-1' restarted successfully; all partitions reassigned."
        ],
        "ERROR": [
            "Kafka internal error: Internal server error occurred while producing message to topic 'orders'; stack trace logged.",
            "Failed to publish message: Failed to publish message to topic 'events' due to broker unavailability; retrying.",
            "Unexpected broker error: Broker 'broker-2' encountered disk I/O error; manual intervention required."
        ]
    }
}

import uuid 
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
    """Connect to Milvus."""
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")

def create_collection(collection_name: str):
    """Create a Milvus collection."""
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
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Embedding size
    ]
    schema = CollectionSchema(fields, description=f"Logs for {collection_name}")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created successfully!")
    return collection

def create_index(collection_name: str):
    """Create an index on the embedding field."""
    collection = Collection(name=collection_name)
    index_params = {
        "index_type": "IVF_FLAT",  # Index type
        "metric_type": "L2",       # Distance metric
        "params": {"nlist": 128}   # Index parameters
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Index created on collection '{collection_name}'.")
# Function to add UUIDs to some log messages

def add_uuids_to_logs(errors_dict):
    for log_type, levels in errors_dict.items():
        for level, messages in levels.items():
            for i in range(len(messages)):
                if random.random() < 0.3:  # 30% chance to add a UUID
                    messages[i] = f"{messages[i]} [{uuid.uuid4()}]"
    return errors_dict

# Add UUIDs to some log messages
#ERRORS = add_uuids_to_logs(ERRORS)

def generate_log(collection_type: str):
    """Generate a single log entry."""
    log_level = random.choices(list(LOG_LEVEL_PROB.keys()), weights=LOG_LEVEL_PROB.values())[0]
    app_category = random.choice(list(APPLICATIONS[collection_type].keys()))
    application = random.choice(APPLICATIONS[collection_type][app_category])
    
    # global ERRORS
    # ERRORS = add_uuids_to_logs(ERRORS)
    error_message = random.choice(ERRORS[collection_type][log_level])
    

    #uuid = uuid.uuid4() 
    
    log_entry = {
        "timestamp": int(time.time() * 1000),  # Unix timestamp in milliseconds
        "level": log_level,
        "application": application,
        "category": app_category,
        "message": f"{error_message} [{uuid.uuid4() }]"
    }
    return log_entry

def generate_log_embedding(log_message):
    """Generate embedding for the log message."""
    return model.encode(log_message).tolist()

def insert_log_to_milvus(log_entry, collection_name):
    """Insert a log entry into Milvus."""
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
    """Continuously generate and stream logs."""
    for _ in range(1000):  # Loop 10 times (adjust as needed)
        log_entry = generate_log(collection_type)  # Generate a log entry
        insert_log_to_milvus(log_entry, collection_name)  # Insert the log into Milvus collection
        #time.sleep(random.uniform(0.5, 2))  # Simulate streaming interval
    print(f"Streamed logs for {collection_type}.")

if __name__ == "__main__":
    MILVUS_HOST = "192.168.2.220"
    MILVUS_PORT = "19530"
    
    # Connect to Milvus
    connect_to_milvus(MILVUS_HOST, MILVUS_PORT)

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
            collection = create_collection(collection_name)
            create_index(collection_name)  # Create index after collection creation
        else:
            print(f"Collection '{collection_name}' already exists.")
    
    # Start streaming logs for each collection
    for collection_name, collection_type in collections.items():
        stream_logs(collection_type, collection_name)