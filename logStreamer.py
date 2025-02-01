import random
import time
import json

# Define log levels and their probability distribution
LOG_LEVELS = {
    "CRITICAL": 10,  # 10% of logs
    "WARNING": 20,   # 20% of logs
    "INFO": 70       # 70% of logs
}

# Normalize probabilities
TOTAL = sum(LOG_LEVELS.values())
LOG_LEVEL_PROB = {level: weight / TOTAL for level, weight in LOG_LEVELS.items()}

# Applications and their types
APPLICATIONS = {
    "frontend": ["ui-service", "web-client"],
    "backend": ["auth-service", "order-service", "payment-gateway"],
    "database": ["mysql-db", "redis-cache"],
    "messaging": ["kafka-broker", "rabbitmq-server"]
}

# Predefined errors for different levels
ERRORS = {
    "CRITICAL": ["Database connection failed", "Application crashed", "Payment gateway down"],
    "WARNING": ["High response time", "Slow query detected", "Memory usage high"],
    "INFO": ["User logged in", "Page loaded successfully", "Cache hit rate optimal"]
}

def generate_log():
    """Generates a single log entry."""
    log_level = random.choices(list(LOG_LEVEL_PROB.keys()), weights=LOG_LEVEL_PROB.values())[0]
    app_category = random.choice(list(APPLICATIONS.keys()))
    application = random.choice(APPLICATIONS[app_category])
    error_message = random.choice(ERRORS[log_level])
    
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": log_level,
        "application": application,
        "category": app_category,
        "message": error_message 
    }
    return json.dumps(log_entry)

def stream_logs():
    """Continuously generates and streams logs."""
    while True:
        print(generate_log())
        time.sleep(random.uniform(0.5, 2))  # Simulate streaming interval

if __name__ == "__main__":
    stream_logs()
