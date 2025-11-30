"""
Test different embedding models
"""
from src_rag import evaluate

EMBEDDING_MODELS = [
    'BAAI/bge-base-en-v1.5',
    'sentence-transformers/all-MiniLM-L6-v2',
    'intfloat/multilingual-e5-large'
]

def test_embeddings():
    results = []
    
    for model in EMBEDDING_MODELS:
        print(f"\nTesting {model}")
        
        config = {
            "model": {
                "chunk_size": 512,
                "chunk_overlap": 50,
                "embedding_model": model
            }
        }
        
        try:
            evaluate.run_evaluate_retrieval(config)
            results.append((model, "OK"))
        except Exception as e:
            print(f"Error: {e}")
            results.append((model, "FAILED"))
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    for model, status in results:
        print(f"{model}: {status}")

if __name__ == "__main__":
    test_embeddings()
