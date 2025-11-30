"""
Test Small2Big strategy
"""
from src_rag import evaluate

def test_small2big():
    configs = [
        {
            "name": "Standard 256",
            "config": {"model": {"chunk_size": 256, "use_small2big": False}}
        },
        {
            "name": "Small2Big 256 ratio=2",
            "config": {"model": {"chunk_size": 256, "use_small2big": True, "small2big_ratio": 2}}
        },
        {
            "name": "Small2Big 256 ratio=3",
            "config": {"model": {"chunk_size": 256, "use_small2big": True, "small2big_ratio": 3}}
        },
    ]
    
    results = []
    
    for test in configs:
        print(f"\nTesting {test['name']}")
        
        try:
            evaluate.run_evaluate_retrieval(test['config'])
            results.append((test['name'], "OK"))
        except Exception as e:
            print(f"Error: {e}")
            results.append((test['name'], "FAILED"))
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    for name, status in results:
        print(f"{name}: {status}")

if __name__ == "__main__":
    test_small2big()
