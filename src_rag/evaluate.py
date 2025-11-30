from pathlib import Path
import mlflow
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from time import sleep
import yaml

from src_rag import models

from FlagEmbedding import FlagModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONF = yaml.safe_load(open("config.yml"))

FOLDER = Path("data") / "raw" / "movies" / "wiki"
FILENAMES = [
    FOLDER / title for title in ["Inception.md", "The Dark Knight.md", "Deadpool.md", "Fight Club.md", "Pulp Fiction.md"]
]
DF = pd.read_csv("data/raw/movies/questions.csv", sep=";") 

ENCODER = SentenceTransformer('all-MiniLM-L6-v2')

def _load_ml_flow(conf):
    mlflow.set_experiment("RAG_Movies_clean")

_load_ml_flow(CONF)

def run_comprehensive_evaluation():
    """Exécute toutes les configurations d'amélioration"""
    
    # Configurations de test
    configurations = [
        # Baseline
        {
            "chunk_size": 256,
        },
        # Améliorations de chunking
        {
            "chunk_size": 256,
            "overlap": 50,
        },
        {
            "chunk_size": 512,
            "overlap": 100,
        },
        # Améliorations d'embedding
        {
            "chunk_size": 256,
            "overlap": 50,
            "embedding_model": "BAAI/bge-large-en-v1.5",
        },
        # Améliorations de retrieval
        {
            "chunk_size": 256,
            "overlap": 50,
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "use_reranking": True,
        },
        {
            "chunk_size": 256,
            "overlap": 50,
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "use_hybrid": True,
        },
        # Configuration optimale
        {
            "chunk_size": 256,
            "overlap": 50,
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "use_reranking": True,
            "use_hybrid": True,
        }
    ]
    
    results = []
    for config in configurations:
        mrr_score = run_evaluate_retrieval({"model": config})
        results.append({
            "mrr": mrr_score
        })
    
    # Afficher les résultats
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("RÉSULTATS COMPARATIFS")
    print("="*50)
    print(results_df.sort_values('mrr', ascending=False))
    
    return results_df

def run_evaluate_retrieval(config, rag=None):
    rag = rag or models.get_model(config)
    # Utiliser .copy() pour éviter les warnings
    df_clean = DF.dropna().copy()
    score = evaluate_retrieval(rag, FILENAMES, df_clean)

    _push_mlflow_result(score, config)
    
    return score["mrr"]  # Retourner le MRR pour comparaison

def run_evaluate_reply(config, rag=None):
    rag = rag or models.get_model(config)
    indexes = range(2, len(DF), 10)
    # Utiliser .copy() pour éviter SettingWithCopyWarning
    df_subset = DF.iloc[indexes].copy()
    score = evaluate_reply(rag, FILENAMES, df_subset)

    _push_mlflow_result(score, config)
    return rag

def _push_mlflow_result(score, config, description=None):
    with mlflow.start_run(description=description):
        df = score.pop("df_result")
        mlflow.log_table(df, artifact_file="df.json")
        mlflow.log_metrics(score)

        config_no_key = {
            key: val for key, val in config.items() if not key.endswith("_key")
        }

        mlflow.log_dict(config_no_key, "config.json")

def evaluate_reply(rag, filenames, df):
    rag.load_files(filenames)

    replies = []
    for question in tqdm(df["question"]):
        replies.append(rag.reply(question))
        # Not to many requests to groq
        sleep(2)

    df["reply"] = replies
    df["sim"] = df.apply(lambda row: calc_semantic_similarity(row["reply"], row["expected_reply"]), axis=1)
    df["is_correct"] = df["sim"] > .7

    return {
        "reply_similarity": df["sim"].mean(),
        "percent_correct": df["is_correct"].mean(),
        "df_result": df[["question", "reply", "expected_reply", "sim", "is_correct"]],
    }

def evaluate_retrieval(rag, filenames, df_question):
    rag.load_files(filenames)
    ranks = []
    for _, row in df_question.iterrows():
        chunks = rag._get_context(row.question)
        try:
            rank = next(i for i, c in enumerate(chunks) if row.text_answering in c)
        except StopIteration:
            rank = len(chunks)  # Pénalité si non trouvé

        ranks.append(rank)
        
    df_question["rank"] = ranks
            
    # MRR calculation - 1/rank, avec rank=0 si non trouvé devient 0
    mrr = np.mean([0 if r >= len(chunks) else 1 / (r + 1) for r in ranks])

    return {
        "mrr": mrr,
        "nb_chunks": len(rag.get_chunks()),
        "df_result": df_question[["question", "text_answering", "rank"]],
    }

def calc_semantic_similarity(generated_answer: str, reference_answer: str) -> float:
    """
    Calculate semantic similarity between generated and reference answers.
    
    Args:
        generated_answer: The answer produced by the RAG system
        reference_answer: The expected or ground-truth answer
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    # Generate embeddings for both texts
    embeddings = ENCODER.encode([generated_answer, reference_answer])
    generated_embedding = embeddings[0].reshape(1, -1)
    reference_embedding = embeddings[1].reshape(1, -1)
    similarity = cosine_similarity(generated_embedding, reference_embedding)[0][0]
    return float(similarity)

if __name__ == "__main__":
    # Pour exécuter toutes les améliorations
    results = run_comprehensive_evaluation()
    print(f"\nMeilleure configuration: {results.loc[results['mrr'].idxmax()]}")