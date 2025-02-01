from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

def connect_to_milvus(host: str, port: str):
    connections.connect(alias="default", host=host, port=port)
    print(f"Connected to Milvus at {host}:{port}")

def create_collection(collection_name: str):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
    ]
    schema = CollectionSchema(fields, description="Test Collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Collection '{collection_name}' created successfully!")

def check_collection_exists(collection_name: str):
    return collection_name in Collection.list_collections()

if __name__ == "__main__":
    print("Hello")
    MILVUS_HOST = "192.168.2.220"
    MILVUS_PORT = "19530"
    COLLECTION_NAME = "test_collection"

    connect_to_milvus(MILVUS_HOST, MILVUS_PORT)
    
    if COLLECTION_NAME not in Collection.list_collections():
        create_collection(COLLECTION_NAME)
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    
    if check_collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists in Milvus.")
    else:
        print(f"Collection '{COLLECTION_NAME}' does NOT exist.")