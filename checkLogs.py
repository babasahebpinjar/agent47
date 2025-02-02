from pymilvus import connections, Collection, utility

def connect_to_milvus(host: str, port: str):
    """Connect to Milvus."""
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")

def print_collection_data(collection_name: str, limit: int = 10):
    """Print data from a Milvus collection."""
    # Load the collection
    collection = Collection(name=collection_name)
    collection.load()

    # Query the collection to retrieve data
    query_results = collection.query(
        expr="",  # Empty expression to retrieve all data
        output_fields=["log_id", "timestamp", "service", "log_level", "raw_log", "embedding"],  # Fields to retrieve
        limit=limit  # Limit the number of records returned
    )

    # Print the retrieved data
    print(f"Data in collection '{collection_name}':")
    for result in query_results:
        print(f"Log ID: {result['log_id']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Service: {result['service']}")
        print(f"Log Level: {result['log_level']}")
        print(f"Raw Log: {result['raw_log']}")
        print(f"Embedding: {result['embedding'][:5]}...")  # Print first 5 elements of the embedding vector
        print("-" * 50)

if __name__ == "__main__":
    MILVUS_HOST = "192.168.2.220"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "httpd_logs"

    # Connect to Milvus
    connect_to_milvus(MILVUS_HOST, MILVUS_PORT)

    # Check if the collection exists
    if COLLECTION_NAME in utility.list_collections():
        # Print data from the collection
        print_collection_data(COLLECTION_NAME, limit=10)  # Adjust the limit as needed
    else:
        print(f"Collection '{COLLECTION_NAME}' does NOT exist.")