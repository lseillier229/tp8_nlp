# 1. Contexte général

Dans le cadre du **TD6**, nous avons développé et optimisé un système **RAG (Retrieval-Augmented Generation)** basé sur un corpus de pages Wikipédia et un ensemble de questions/réponses de référence.  
L’objectif est de **maximiser le MRR (Mean Reciprocal Rank)** en optimisant le *chunking*, les *embeddings* et le *retrieval*, tout en conservant une architecture simple.

Le groupe s’est organisé en parallèle : chaque membre explorait une piste d’amélioration différente, puis nous avons fusionné les résultats dans un dépôt Git commun.

---

# 2. Notre démarche globale

Notre stratégie s’est déroulée en **quatre phases collaboratives**, où chacun testait une dimension du RAG.

## Phase 1 — Reproduction du baseline

Nous avons d’abord exécuté le code fourni (`config.yml` + `evaluate.run_evaluate_retrieval`).

**Résultat baseline moyen :**

- MRR ≈ **0.18**

Cela nous a servi de point de départ.

---

## Phase 2 — Optimisation du chunking

### Ce que nous avons testé

- **Tailles de chunks :**  
  - 128  
  - 256  
  - 384  
  - 512  
  - 768  

- **Différents overlaps :**  
  - 0 %  
  - 25 %  
  - 50 %  
  - 75 %  

- **Différentes stratégies de découpage :**
  - découpage naïf par tokens  
  - *sliding window*  
  - *small2big*  
  - hiérarchisation via les headers Markdown  
  - regroupement par paragraphes  
  - chunking multi-granulaire  

### Ce que nous avons observé

- Les petits chunks (**128–256**) → MRR plus faible, car trop fragmentés.
- Les chunks plus longs (**512–768**) → meilleurs scores, car plus de contexte.
- L’**overlap** améliore systématiquement le MRR :  
  → il permet d’éviter la perte d’information en bordure de chunk.

---

## Phase 3 — Optimisation des embeddings & du retrieval

### Normalisation des embeddings

Nous avons appliqué une **normalisation L2** des embeddings, de façon à se rapprocher d’une similarité cosinus lors du calcul des scores.

### Modèles d’embeddings testés

- `BAAI/bge-base-en-v1.5`
- `BAAI/bge-large-en-v1.5`
- `all-MiniLM-L6-v2` (pour la similarité de réponse)

Ces changements ont apporté un **boost modéré mais perceptible** sur le MRR (environ **+5 à +10 %** selon les configurations).

---

## Phase 4 — Mutualisation & meilleure configuration du groupe

En fusionnant les contributions (choix de `chunk_size`, `chunk_overlap`, modèles d’embeddings plus gros, hiérarchisation Markdown, *small2big*), une configuration du groupe a atteint :

- **MRR ≈ 0.48** (meilleure configuration du groupe)

### Cette configuration combine notamment :

- `chunk_size = 512`
- `chunk_overlap = 128`
- `embed_model = "BAAI/bge-large-en-v1.5"`
- chunking **multi-granulaire / small2big**
- retrieval **hybride** avec un léger **reranking**
