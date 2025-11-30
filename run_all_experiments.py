"""
Run all experiments and log to MLflow
"""
from src_rag import evaluate

def run_all_experiments():
    experiments = []
    
    # Chunking experiments
    for chunk_size in [256, 512, 1024]:
        for overlap in [0, 50, 100]:
            if overlap >= chunk_size:
                continue
            experiments.append({
                "chunk_size": chunk_size,
                "chunk_overlap": overlap
            })
    
    # Small2Big experiments
    for ratio in [2, 3]:
        experiments.append({
            "chunk_size": 256,
            "use_small2big": True,
            "small2big_ratio": ratio
        })
    
    # Embedding experiments
    for model in ['BAAI/bge-base-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'intfloat/multilingual-e5-large']:
        experiments.append({
            "chunk_size": 512,
            "chunk_overlap": 50,
            "embedding_model": model
        })
    
    total = len(experiments)
    print(f"Running {total} experiments\n")
    
    for i, exp_config in enumerate(experiments, 1):
        print(f"[{i}/{total}] Testing {exp_config}")
        
        config = {"model": exp_config}
        
        try:
            evaluate.run_evaluate_retrieval(config)
        except Exception as e:
            print(f"Error: {e}")
    
    print(f"\nCompleted {total} experiments")
    print("Check results in MLflow: mlflow ui")

if __name__ == "__main__":
    run_all_experiments()
