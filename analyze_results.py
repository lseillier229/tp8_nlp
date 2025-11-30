"""
Analyze MLflow results and compare configurations
"""
import mlflow
import pandas as pd
import json

def analyze_results():
    try:
        experiment = mlflow.get_experiment_by_name("RAG_Movies_clean")
        if experiment is None:
            print("No experiments found. Run experiments first.")
            return
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.mrr DESC"]
        )
        
        if len(runs) == 0:
            print("No runs found")
            return
        
        print(f"\nTotal runs: {len(runs)}\n")
        
        print("="*100)
        print("TOP 10 CONFIGURATIONS BY MRR")
        print("="*100)
        print(f"{'Rank':<6} {'MRR':<8} {'Chunks':<8} {'Config':<75}")
        print("-"*100)
        
        for idx, (_, run) in enumerate(runs.head(10).iterrows(), 1):
            mrr = run.get('metrics.mrr', 0)
            nb_chunks = int(run.get('metrics.nb_chunks', 0))
            
            # Extract config
            config_str = ""
            try:
                config_json = run.get('params.config.json', '{}')
                if config_json and config_json != '{}':
                    config = json.loads(config_json)
                    chunk_size = config.get('chunk_size', 'N/A')
                    overlap = config.get('chunk_overlap', 0)
                    embed = config.get('embedding_model', 'default').split('/')[-1]
                    s2b = config.get('use_small2big', False)
                    ratio = config.get('small2big_ratio', 0)
                    
                    config_str = f"chunk={chunk_size} ovlp={overlap} embed={embed}"
                    if s2b:
                        config_str += f" s2b=r{ratio}"
            except:
                config_str = "N/A"
            
            print(f"{idx:<6} {mrr:<8.4f} {nb_chunks:<8} {config_str:<75}")
        
        print("\n" + "="*100)
        print("STATISTICS")
        print("="*100)
        print(f"Best MRR:    {runs['metrics.mrr'].max():.4f}")
        print(f"Average MRR: {runs['metrics.mrr'].mean():.4f}")
        print(f"Worst MRR:   {runs['metrics.mrr'].min():.4f}")
        
        # Best configuration details
        best_run = runs.iloc[0]
        print("\n" + "="*100)
        print("BEST CONFIGURATION DETAILS")
        print("="*100)
        try:
            config_json = best_run.get('params.config.json', '{}')
            if config_json and config_json != '{}':
                config = json.loads(config_json)
                for key, value in config.items():
                    print(f"{key}: {value}")
        except:
            print("Config not available")
        
        print("\n" + "="*100)
        print("NEXT STEPS")
        print("="*100)
        print("1. Review top configurations above")
        print("2. Launch MLflow UI: uv run mlflow ui")
        print("3. Test best configuration on reply task")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_results()
