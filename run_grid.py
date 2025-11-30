from src_rag import evaluate

# --- Grille d'hyperparamètres à tester ---

# tailles de chunks
CHUNK_SIZES = [ 256, 384]

# overlap en fonction du chunk_size (pour éviter les combos débiles)
OVERLAPS_BY_CS = {
    256: [0, 64, 128],     # 0, 25%, 50%
    384: [0, 96, 192],     # 0, 25%, 50%
}

# nombre de chunks utilisés pour répondre
TOP_KS = [5, 8]

# modèles d'embeddings à comparer
EMBED_MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
]


def main():
    run_id = 0

    for cs in CHUNK_SIZES:
        overlaps = OVERLAPS_BY_CS.get(cs, [0])

        for ov in overlaps:
            for tk in TOP_KS:
                for em in EMBED_MODELS:
                    run_id += 1
                    config = {
                        "model": {
                            "chunk_size": cs,
                            "chunk_overlap": ov,
                            "top_k": tk,
                            "embed_model": em,
                        }
                    }

                    print(f"\n===== RUN {run_id} =====")
                    print(f"chunk_size   = {cs}")
                    print(f"chunk_overlap= {ov}")
                    print(f"top_k        = {tk}")
                    print(f"embed_model  = {em}")

                    # lance l'évaluation (log automatiquement dans MLflow)
                    evaluate.run_evaluate_retrieval(config)


if __name__ == "__main__":
    main()
