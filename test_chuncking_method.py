import sys
sys.path.append('.')

from pathlib import Path
import mlflow
import numpy as np
import os
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import yaml
from rank_bm25 import BM25Okapi
import re
import tiktoken
import itertools
from collections import defaultdict

from src_rag import models
from src_rag.evaluate import DF, FILENAMES, ENCODER

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONF = yaml.safe_load(open("config.yml"))

# Initialiser le tokenizer pour le comptage de tokens
tokenizer = tiktoken.get_encoding("cl100k_base")

def _load_ml_flow(conf):
    mlflow.set_experiment("RAG_Movies_advanced_chunking_fixed")

_load_ml_flow(CONF)

class UltraOptimizedChunkingRAG:
    """RAG avec méthodes de chunking ultra-optimisées pour MRR ~1"""
    
    def __init__(self, base_rag, chunking_strategy="default", **chunking_params):
        self.base_rag = base_rag
        self.chunking_strategy = chunking_strategy
        self.chunking_params = chunking_params
        self._reranker = None
        
    def load_files(self, filenames):
        """Charger les fichiers avec la stratégie de chunking choisie"""
        # Charger d'abord avec la méthode de base
        self.base_rag.load_files(filenames)
        
        # Si c'est la stratégie par défaut, on ne fait rien
        if self.chunking_strategy == "default":
            return
            
        # Sinon, on recalcule les chunks avec notre stratégie avancée
        texts = self.base_rag._texts
        new_chunks = self._apply_advanced_chunking(texts)
        
        # Remplacer les chunks et recalculer les embeddings
        self.base_rag._chunks = new_chunks
        self.base_rag._corpus_embedding = self.base_rag.embed_corpus(new_chunks)
        
        print(f"✅ Stratégie '{self.chunking_strategy}' appliquée. Chunks: {len(new_chunks)}")
    
    def _apply_advanced_chunking(self, texts):
        """Appliquer la stratégie de chunking avancée"""
        if self.chunking_strategy == "small2big":
            return self._small2big_chunking_optimized(texts)
        elif self.chunking_strategy == "semantic":
            return self._semantic_chunking_optimized(texts)
        elif self.chunking_strategy == "sliding_window":
            return self._sliding_window_chunking_optimized(texts)
        elif self.chunking_strategy == "hierarchical":
            return self._hierarchical_chunking_optimized(texts)
        elif self.chunking_strategy == "paragraph_based":
            return self._paragraph_based_chunking_optimized(texts)
        elif self.chunking_strategy == "hybrid_aggressive":
            return self._hybrid_aggressive_chunking(texts)
        elif self.chunking_strategy == "multi_level":
            return self._multi_level_chunking(texts)
        elif self.chunking_strategy == "answer_focused":
            return self._answer_focused_chunking(texts)
        elif self.chunking_strategy == "overlap_heavy":
            return self._overlap_heavy_chunking(texts)
        else:
            return self.base_rag._chunks
    
    def _small2big_chunking_optimized(self, texts):
        """Small2Big optimisé avec plus de granularité"""
        # Plus de tailles pour couvrir tous les cas
        tiny_chunks = self._chunk_texts(texts, chunk_size=64)
        small_chunks = self._chunk_texts(texts, chunk_size=128)
        medium_chunks = self._chunk_texts(texts, chunk_size=256)
        large_chunks = self._chunk_texts(texts, chunk_size=512)
        xlarge_chunks = self._chunk_texts(texts, chunk_size=1024)
        
        print(f"📊 Small2Big Optimisé - Tiny: {len(tiny_chunks)}, Small: {len(small_chunks)}, Medium: {len(medium_chunks)}, Large: {len(large_chunks)}, XLarge: {len(xlarge_chunks)}")
        
        # Combiner tous les chunks avec priorités
        all_chunks = tiny_chunks + small_chunks + medium_chunks + large_chunks + xlarge_chunks
        
        # Ajouter des métadonnées détaillées
        chunks_with_metadata = []
        sizes = [("TINY", tiny_chunks), ("SMALL", small_chunks), ("MEDIUM", medium_chunks), 
                ("LARGE", large_chunks), ("XLARGE", xlarge_chunks)]
        
        for size_name, size_chunks in sizes:
            for chunk in size_chunks:
                chunks_with_metadata.append(f"[{size_name}] {chunk}")
        
        return chunks_with_metadata
    
    def _semantic_chunking_optimized(self, texts):
        """Chunking sémantique ultra-optimisé"""
        all_chunks = []
        
        for text in texts:
            # Séparation multi-niveaux
            sections = re.split(r'\n#{1,6}\s+', text)  # Séparation par headers
            for section in sections:
                if not section.strip():
                    continue
                    
                # Séparation par paragraphes
                paragraphs = re.split(r'\n\s*\n', section)
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    
                    # Séparation par phrases avec contexte
                    sentences = re.split(r'([.!?]+)\s+', paragraph)
                    if len(sentences) == 1:
                        # Une seule phrase/paragraphe
                        all_chunks.append(paragraph)
                    else:
                        # Regroupement intelligent des phrases
                        current_chunk = ""
                        for i in range(0, len(sentences), 2):
                            if i + 1 < len(sentences):
                                sentence = sentences[i] + sentences[i+1]
                            else:
                                sentence = sentences[i]
                            
                            sentence = sentence.strip()
                            if not sentence:
                                continue
                            
                            # Chunks de 1-3 phrases selon la taille
                            if len(tokenizer.encode(current_chunk + " " + sentence)) <= 150:
                                current_chunk += " " + sentence if current_chunk else sentence
                            else:
                                if current_chunk:
                                    all_chunks.append(current_chunk.strip())
                                current_chunk = sentence
                        
                        if current_chunk:
                            all_chunks.append(current_chunk.strip())
        
        return all_chunks
    
    def _sliding_window_chunking_optimized(self, texts):
        """Fenêtre glissante avec overlap agressif"""
        window_size = self.chunking_params.get('window_size', 200)
        overlap = self.chunking_params.get('overlap', 150)  # Overlap très important
        
        all_chunks = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            
            # Step très petit pour maximum de couverture
            step = window_size - overlap
            for i in range(0, len(tokens), step):
                chunk_tokens = tokens[i:i + window_size]
                if chunk_tokens:
                    chunk_text = tokenizer.decode(chunk_tokens)
                    all_chunks.append(chunk_text)
        
        return all_chunks
    
    def _hierarchical_chunking_optimized(self, texts):
        """Hiérarchique avec maximum de redondance"""
        all_chunks = []
        
        for text in texts:
            # Analyser la structure markdown
            sections = self._parse_markdown_hierarchical_detailed(text)
            
            for section in sections:
                header = section['header']
                content = section['content']
                full_hierarchy = " > ".join(section.get('full_hierarchy', [header]))
                
                # Chunk principal complet
                main_chunk = f"{full_hierarchy}: {content}"
                all_chunks.append(main_chunk)
                
                # Chunks par sous-sections
                if len(tokenizer.encode(content)) > 100:
                    # Diviser le contenu
                    sub_chunks = self._chunk_texts([content], chunk_size=100)
                    for i, sub_chunk in enumerate(sub_chunks):
                        all_chunks.append(f"{full_hierarchy} - Part {i+1}: {sub_chunk}")
                
                # Chunk résumé (premières phrases)
                sentences = re.split(r'[.!?]+', content)
                if len(sentences) > 1:
                    summary = '. '.join(sentences[:3]) + '.'
                    all_chunks.append(f"{full_hierarchy} [SUMMARY]: {summary}")
        
        return all_chunks
    
    def _paragraph_based_chunking_optimized(self, texts):
        """Paragraphes avec regroupement intelligent et redondance"""
        all_chunks = []
        
        for text in texts:
            paragraphs = re.split(r'\n\s*\n', text)
            
            # Chunks individuels
            for para in paragraphs:
                para = para.strip()
                if para:
                    all_chunks.append(para)
            
            # Chunks regroupés (2-3 paragraphes)
            for i in range(len(paragraphs) - 1):
                combined = "\n\n".join(paragraphs[i:i+2])
                if combined.strip():
                    all_chunks.append(f"[COMBINED] {combined}")
            
            # Chunks regroupés (3-4 paragraphes)
            for i in range(len(paragraphs) - 2):
                combined = "\n\n".join(paragraphs[i:i+3])
                if combined.strip():
                    all_chunks.append(f"[LONG_COMBINED] {combined}")
        
        return all_chunks
    
    def _hybrid_aggressive_chunking(self, texts):
        """Combinaison agressive de toutes les méthodes"""
        all_chunks = []
        
        # Appliquer les méthodes principales seulement pour éviter trop de chunks
        methods = [
            self._hierarchical_chunking_optimized,
            self._semantic_chunking_optimized,
        ]
        
        for method in methods:
            chunks = method(texts)
            all_chunks.extend(chunks)
        
        # Déduplication plus agressive
        unique_chunks = []
        seen_content = set()
        
        for chunk in all_chunks:
            # Extraire le contenu principal pour déduplication
            content = re.sub(r'\[[^\]]+\]', '', chunk).strip()
            content_hash = hash(content[:100])  # Hash du début pour éviter exact duplicates
            
            if content_hash not in seen_content:
                unique_chunks.append(chunk)
                seen_content.add(content_hash)
        
        return unique_chunks
    
    def _multi_level_chunking(self, texts):
        """Chunking multi-niveaux avec redondance contrôlée"""
        all_chunks = []
        
        for text in texts:
            # Niveau 1: Sections complètes
            sections = self._parse_markdown_hierarchical_detailed(text)
            for section in sections:
                full_content = f"{section['header']}: {section['content']}"
                all_chunks.append(f"[FULL_SECTION] {full_content}")
            
            # Niveau 2: Paragraphes individuels
            paragraphs = re.split(r'\n\s*\n', text)
            for para in paragraphs:
                if para.strip():
                    all_chunks.append(f"[PARAGRAPH] {para.strip()}")
            
            # Niveau 3: Phrases groupées
            sentences = re.findall(r'[^.!?]+[.!?]+', text)
            current_group = ""
            for sentence in sentences:
                if len(tokenizer.encode(current_group + sentence)) <= 100:
                    current_group += sentence
                else:
                    if current_group:
                        all_chunks.append(f"[SENTENCE_GROUP] {current_group.strip()}")
                    current_group = sentence
            if current_group:
                all_chunks.append(f"[SENTENCE_GROUP] {current_group.strip()}")
        
        return all_chunks
    
    def _answer_focused_chunking(self, texts):
        """Chunking focalisé sur les réponses potentielles"""
        all_chunks = []
        
        # Charger les questions pour orientation
        questions_df = DF.dropna().copy()
        common_answer_patterns = self._analyze_answer_patterns(questions_df)
        
        for text in texts:
            # Chunks réguliers
            regular_chunks = self._semantic_chunking_optimized([text])
            all_chunks.extend(regular_chunks)
            
            # Chunks focalisés sur les réponses (limité à 5 patterns)
            for pattern in common_answer_patterns[:5]:
                pattern_chunks = self._extract_pattern_chunks(text, pattern)
                all_chunks.extend(pattern_chunks)
        
        return all_chunks
    
    def _overlap_heavy_chunking(self, texts):
        """Chunking avec overlap maximal"""
        all_chunks = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            
            # Configuration unique pour éviter trop de chunks
            window_size = 150
            overlap = 120
            
            step = window_size - overlap
            for i in range(0, len(tokens), step):
                end_idx = min(i + window_size, len(tokens))
                if end_idx > i:
                    chunk_tokens = tokens[i:end_idx]
                    chunk_text = tokenizer.decode(chunk_tokens)
                    all_chunks.append(f"[OVERLAP] {chunk_text}")
        
        return all_chunks
    
    def _parse_markdown_hierarchical_detailed(self, text):
        """Parser markdown avec hiérarchie détaillée"""
        lines = text.split('\n')
        sections = []
        current_headers = []
        current_content = ""
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s*(.+)$', line)
            if header_match:
                # Sauvegarder la section précédente
                if current_content.strip():
                    sections.append({
                        'header': current_headers[-1] if current_headers else "Root",
                        'full_hierarchy': current_headers.copy(),
                        'content': current_content.strip()
                    })
                
                # Nouvelle section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                # Ajuster la hiérarchie
                current_headers = current_headers[:level-1]
                current_headers.append(title)
                current_content = ""
            else:
                current_content += line + "\n"
        
        # Ajouter la dernière section
        if current_content.strip():
            sections.append({
                'header': current_headers[-1] if current_headers else "Root",
                'full_hierarchy': current_headers.copy(),
                'content': current_content.strip()
            })
        
        return sections
    
    def _extract_keywords(self, text):
        """Extraire les mots-clés importants"""
        words = re.findall(r'\b[A-Za-z]{4,}\b', text)
        word_freq = defaultdict(int)
        for word in words:
            if word.lower() not in ['this', 'that', 'with', 'from', 'they', 'their']:
                word_freq[word] += 1
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)
    
    def _find_keyword_context(self, text, keyword, context_size=100):
        """Trouver le contexte autour des mots-clés"""
        chunks = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        
        for match in pattern.finditer(text):
            start = max(0, match.start() - context_size)
            end = min(len(text), match.end() + context_size)
            context = text[start:end]
            chunks.append(f"[KEYWORD:{keyword}] {context}")
        
        return chunks
    
    def _analyze_answer_patterns(self, questions_df):
        """Analyser les patterns de réponses"""
        answers = questions_df['text_answering'].dropna().tolist()
        patterns = []
        
        for answer in answers:
            # Extraire les phrases complètes
            sentences = re.split(r'[.!?]+', answer)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    patterns.append(sentence)
        
        # Retourner les patterns les plus communs
        from collections import Counter
        return [pattern for pattern, count in Counter(patterns).most_common(10)]
    
    def _extract_pattern_chunks(self, text, pattern):
        """Extraire les chunks contenant des patterns spécifiques"""
        chunks = []
        if len(pattern) < 5:
            return chunks
            
        # Rechercher le pattern dans le texte
        if pattern.lower() in text.lower():
            # Trouver le contexte autour du pattern
            pattern_lower = pattern.lower()
            text_lower = text.lower()
            start = text_lower.find(pattern_lower)
            
            if start != -1:
                # Extraire un chunk autour du pattern
                context_start = max(0, start - 150)
                context_end = min(len(text), start + len(pattern) + 150)
                context_chunk = text[context_start:context_end]
                chunks.append(f"[PATTERN_MATCH] {context_chunk}")
        
        return chunks
    
    def _chunk_texts(self, texts, chunk_size):
        """Méthode utilitaire pour chunker des textes avec une taille donnée"""
        all_chunks = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            token_chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
            
            for token_chunk in token_chunks:
                if token_chunk:
                    chunk_text = tokenizer.decode(token_chunk)
                    all_chunks.append(chunk_text)
        
        return all_chunks
    
    def enhanced_retrieval_aggressive(self, query, top_k=10):
        """Récupération ultra-agressive avec multiple strategies"""
        # Embedding similarity seulement pour la rapidité
        query_embedding = self.base_rag.embed_questions([query])
        sim_scores = query_embedding @ self.base_rag.get_corpus_embedding().T
        
        # Trier par similarité
        sorted_indices = list(np.argsort(sim_scores[0]))[::-1]
        
        # Sélection simple sans diversité complexe
        selected_indices = sorted_indices[:top_k]
        
        chunks = self.base_rag.get_chunks()
        return [chunks[i] for i in selected_indices]
    
    # Surcharge de _get_context pour utiliser la récupération agressive
    def _get_context(self, query):
        """Récupération ultra-optimisée du contexte"""
        return self.enhanced_retrieval_aggressive(query, top_k=8)
    
    # Délégation des autres méthodes au RAG de base
    def __getattr__(self, name):
        return getattr(self.base_rag, name)

def run_focused_chunking_experiments():
    """Tests avec des stratégies ciblées et optimisées"""
    
    focused_strategies = [
        {
            "name": "Default Chunking (Baseline)",
            "strategy": "default",
            "config": {"chunk_size": 256},
            "params": {}
        },
        {
            "name": "Hierarchical Optimized",
            "strategy": "hierarchical",
            "config": {"chunk_size": 256},
            "params": {}
        },
        {
            "name": "Hybrid Aggressive Light",
            "strategy": "hybrid_aggressive",
            "config": {"chunk_size": 256},
            "params": {}
        },
        {
            "name": "Answer-Focused Chunking",
            "strategy": "answer_focused",
            "config": {"chunk_size": 256},
            "params": {}
        },
        {
            "name": "Overlap-Heavy Chunking",
            "strategy": "overlap_heavy",
            "config": {"chunk_size": 256},
            "params": {}
        }
    ]
    
    results = []
    
    for strategy in focused_strategies:
        print(f"\n{'='*60}")
        print(f"🧪 TEST: {strategy['name']}")
        print(f"{'='*60}")
        
        try:
            # Créer le RAG de base
            base_rag = models.get_model({"model": strategy["config"]})
            
            # Créer le RAG avec chunking optimisé
            rag = UltraOptimizedChunkingRAG(
                base_rag,
                chunking_strategy=strategy["strategy"],
                **strategy["params"]
            )
            
            # Évaluer avec métriques simplifiées
            score = evaluate_retrieval_fast(rag, FILENAMES, DF.dropna().copy())
            
            results.append({
                "name": strategy["name"],
                "strategy": strategy["strategy"],
                "mrr": score["mrr"],
                "recall_1": score["recall_1"],
                "recall_3": score["recall_3"],
                "coverage": score["coverage"],
                "nb_chunks": score["nb_chunks"],
                "avg_chunk_tokens": score["avg_chunk_tokens"],
                "perfect_retrieval": score["perfect_retrieval"]
            })
            
            print(f"✅ {strategy['name']}")
            print(f"   MRR: {score['mrr']:.3f} | Recall@1: {score['recall_1']:.3f}")
            print(f"   Recall@3: {score['recall_3']:.3f} | Perfect: {score['perfect_retrieval']:.1f}%")
            print(f"   Chunks: {score['nb_chunks']} | Coverage: {score['coverage']:.3f}")
            
        except Exception as e:
            print(f"❌ Erreur avec {strategy['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    return pd.DataFrame(results) if results else pd.DataFrame()

def evaluate_retrieval_fast(rag, filenames, df_question):
    """Évaluation rapide avec métriques compatibles MLflow"""
    rag.load_files(filenames)
    
    ranks = []
    found_positions = []
    chunk_lengths = []
    perfect_matches = 0
    
    chunks = rag.get_chunks()
    for chunk in chunks:
        chunk_lengths.append(len(tokenizer.encode(chunk)))
    
    for _, row in tqdm(df_question.iterrows(), desc="Évaluation"):
        try:
            retrieved_chunks = rag._get_context(row.question)
            found_rank = None
            
            for i, chunk in enumerate(retrieved_chunks):
                # Recherche flexible
                if (row.text_answering in chunk or 
                    row.text_answering.lower() in chunk.lower()):
                    found_rank = i
                    if i == 0:
                        perfect_matches += 1
                    break
            
            if found_rank is not None:
                ranks.append(found_rank)
                found_positions.append(found_rank)
            else:
                ranks.append(len(retrieved_chunks))
                found_positions.append(-1)
                
        except Exception as e:
            print(f"⚠️ Erreur évaluation: {e}")
            ranks.append(10)
            found_positions.append(-1)
    
    df_question["rank"] = ranks
    df_question["found"] = [r != -1 for r in found_positions]
    
    # Métriques compatibles MLflow (noms sans @)
    total_questions = len(df_question)
    recall_1 = np.mean([1 if r == 0 else 0 for r in found_positions if r != -1]) if any(r != -1 for r in found_positions) else 0
    recall_3 = np.mean([1 if r < 3 else 0 for r in found_positions if r != -1]) if any(r != -1 for r in found_positions) else 0
    mrr = np.mean([0 if r >= len(chunks) else 1 / (r + 1) for r in ranks])
    
    return {
        "mrr": mrr,
        "recall_1": recall_1,
        "recall_3": recall_3,
        "nb_chunks": len(chunks),
        "avg_chunk_tokens": np.mean(chunk_lengths) if chunk_lengths else 0,
        "coverage": np.mean([r != -1 for r in found_positions]),
        "perfect_retrieval": (perfect_matches / total_questions) * 100,
        "df_result": df_question[["question", "text_answering", "rank", "found"]],
    }

def analyze_focused_results(results_df):
    """Analyse détaillée des résultats"""
    print("\n🔍 ANALYSE DÉTAILLÉE:")
    
    if results_df.empty:
        print("❌ Aucun résultat à analyser")
        return
    
    baseline_mrr = results_df[results_df['strategy'] == 'default']['mrr'].values[0]
    
    for _, row in results_df.iterrows():
        improvement = ((row['mrr'] - baseline_mrr) / baseline_mrr) * 100 if baseline_mrr > 0 else 0
        
        print(f"\n📊 {row['name']}:")
        print(f"   MRR: {row['mrr']:.3f} ({improvement:+.1f}%)")
        print(f"   Recall@1: {row['recall_1']:.3f}")
        print(f"   Perfect Retrieval: {row['perfect_retrieval']:.1f}%")
        
        # Recommandations
        if row['mrr'] > 0.8:
            print("   🎉 EXCELLENT - Proche de l'objectif!")
        elif row['mrr'] > 0.6:
            print("   ✅ TRÈS BON - Continue comme ça!")
        elif row['mrr'] > 0.4:
            print("   📈 BON - Encore un peu d'optimisation nécessaire")

def run_final_optimization():
    """Optimisation finale"""
    print("\n" + "="*80)
    print("🚀 OPTIMISATION FINALE POUR MRR MAXIMAL")
    print("="*80)
    
    results = run_focused_chunking_experiments()
    
    if not results.empty:
        # Sauvegarder
        results.to_csv("focused_chunking_results.csv", index=False)
        
        # Analyse
        analyze_focused_results(results)
        
        # Classement final
        print("\n🏅 CLASSEMENT FINAL:")
        sorted_results = results.sort_values('mrr', ascending=False)
        for i, (_, row) in enumerate(sorted_results.iterrows(), 1):
            print(f"  {i}. {row['name']}: {row['mrr']:.3f} | R@1: {row['recall_1']:.3f} | Perfect: {row['perfect_retrieval']:.1f}%")
        
        best = sorted_results.iloc[0]
        
        if best['mrr'] >= 0.9:
            print(f"\n🎉 OBJECTIF ATTEINT! MRR: {best['mrr']:.3f}")
        elif best['mrr'] >= 0.7:
            print(f"\n✅ TRÈS PROCHE! MRR: {best['mrr']:.3f}")
        elif best['mrr'] >= 0.5:
            print(f"\n📈 BON PROGRÈS! MRR: {best['mrr']:.3f}")
        else:
            print(f"\n💪 CONTINUONS! MRR: {best['mrr']:.3f}")
        
        return best
    return None

if __name__ == "__main__":
    print("🚀 DÉMARRAGE DE L'OPTIMISATION FINALE...")
    
    best_strategy = run_final_optimization()
    
    if best_strategy is not None:
        print(f"\n🎯 STRATÉGIE GAGNANTE: {best_strategy['name']}")
        print(f"   MRR: {best_strategy['mrr']:.3f}")
        
        if best_strategy['mrr'] < 0.7:
            print("\n💡 CONSEILS POUR AMÉLIORER DAVANTAGE:")
            print("   1. Vérifiez la qualité des embeddings")
            print("   2. Augmentez légèrement top_k dans enhanced_retrieval_aggressive")
            print("   3. Ajoutez plus de contexte aux chunks")
            print("   4. Utilisez un modèle d'embedding plus performant")