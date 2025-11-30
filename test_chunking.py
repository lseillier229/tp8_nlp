"""
Test different chunking configurations
"""
from src_rag import evaluate

CHUNK_SIZES = [256, 512, 1024]
OVERLAPS = [0, 50, 100]

def test_chunking():
    results = []
    
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            if overlap >= chunk_size:
                continue
            
            print(f"\nTesting chunk_size={chunk_size}, overlap={overlap}")
            
            config = {
                "model": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap
                }
            }
            
            try:
                evaluate.run_evaluate_retrieval(config)
                results.append((chunk_size, overlap, "OK"))
            except Exception as e:
                print(f"Error: {e}")
                results.append((chunk_size, overlap, "FAILED"))
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    for chunk_size, overlap, status in results:
        print(f"chunk_size={chunk_size:4d}, overlap={overlap:3d}: {status}")

if __name__ == "__main__":
    test_chunking()
