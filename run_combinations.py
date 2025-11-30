"""
Run all possible combinations of RAG parameters
"""
from itertools import product
from src_rag import evaluate

CHUNK_SIZES = [256, 512, 1024]
OVERLAPS = [0, 50, 100]
EMBEDDINGS = [
    'BAAI/bge-base-en-v1.5',
    'sentence-transformers/all-MiniLM-L6-v2',
    'intfloat/multilingual-e5-large'
]
SMALL2BIG = [False, True]
SMALL2BIG_RATIOS = [2, 3]

def run_combinations():
    experiments = []
    
    # Generate all combinations
    for chunk_size, overlap, embedding, use_s2b in product(CHUNK_SIZES, OVERLAPS, EMBEDDINGS, SMALL2BIG):
        if overlap >= chunk_size:
            continue
        
        if use_s2b:
            for ratio in SMALL2BIG_RATIOS:
                experiments.append({
                    "chunk_size": chunk_size,
                    "chunk_overlap": overlap,
                    "embedding_model": embedding,
                    "use_small2big": True,
                    "small2big_ratio": ratio
                })
        else:
            experiments.append({
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "embedding_model": embedding,
                "use_small2big": False
            })
    
    total = len(experiments)
    print(f"Total combinations to test: {total}\n")
    
    for i, exp_config in enumerate(experiments, 1):
        config_str = f"chunk={exp_config['chunk_size']}, overlap={exp_config['chunk_overlap']}"
        config_str += f", embed={exp_config['embedding_model'].split('/')[-1]}"
        if exp_config['use_small2big']:
            config_str += f", s2b=ratio{exp_config.get('small2big_ratio', 2)}"
        
        print(f"[{i}/{total}] {config_str}")
        
        config = {"model": exp_config}
        
        try:
            evaluate.run_evaluate_retrieval(config)
        except Exception as e:
            print(f"Error: {e}\n")
    
    print(f"\nCompleted {total} experiments")
    print("Run: uv run python analyze_results.py")

if __name__ == "__main__":
    run_combinations()
