import pandas as pd
import numpy as np
import re
import os
import platform
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import ollama
import os

class Summarizer:
    def __init__(self, ollama_model="gemma3:27b"):
        self.okt = Okt()
        self.vectorizer = TfidfVectorizer()
        self.ollama_model = os.getenv("OLLAMA_MODEL", ollama_model)

    def get_nouns(self, text):
        """명사 추출 및 전처리"""
        if not isinstance(text, str): return ""
        nouns = self.okt.nouns(text)
        return " ".join([n for n in nouns if len(n) > 1])

    def split_sentences(self, text):
        """문장 분리"""
        sentences = re.split(r'(?<=[.!?])\s+', str(text).strip())
        return [s for s in sentences if len(s.strip()) > 5]

    # --- 1. TF-IDF 기반 요약 ---
    def tfidf_summary(self, text, top_n=5):
        sentences = self.split_sentences(text)
        if len(sentences) <= top_n: return "\n\n".join(sentences)
        sent_nouns = [self.get_nouns(s) for s in sentences]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sent_nouns)
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            top_indices = scores.argsort()[-top_n:][::-1]
            top_indices.sort()
            return "\n\n".join([sentences[i] for i in top_indices])
        except: return "\n\n".join(sentences[:top_n])

    # --- 2. TextRank 기반 요약 ---
    def textrank_summary(self, text, top_n=5):
        sentences = self.split_sentences(text)
        if len(sentences) <= top_n: return "\n\n".join(sentences)
        sent_nouns = [self.get_nouns(s) for s in sentences]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sent_nouns)
            sim_mat = (tfidf_matrix * tfidf_matrix.T).toarray()
            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            top_indices = sorted(scores, key=scores.get, reverse=True)[:top_n]
            top_indices.sort()
            return "\n\n".join([sentences[i] for i in top_indices])
        except: return "\n\n".join(sentences[:top_n])

    # --- 3. LSA (잠재 의미 분석) 기반 요약 ---
    def lsa_summary(self, text, top_n=5):
        sentences = self.split_sentences(text)
        if len(sentences) <= top_n: return "\n\n".join(sentences)
        sent_nouns = [self.get_nouns(s) for s in sentences]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sent_nouns)
            svd = TruncatedSVD(n_components=1)
            svd.fit(tfidf_matrix)
            scores = np.abs(svd.components_ @ tfidf_matrix.T.toarray()).flatten()
            top_indices = scores.argsort()[-top_n:][::-1]
            top_indices.sort()
            return "\n\n".join([sentences[i] for i in top_indices])
        except: return "\n\n".join(sentences[:top_n])

    # --- 4. LexRank 기반 요약 ---
    def lexrank_summary(self, text, top_n=5, threshold=0.1):
        sentences = self.split_sentences(text)
        if len(sentences) <= top_n: return "\n\n".join(sentences)
        sent_nouns = [self.get_nouns(s) for s in sentences]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sent_nouns)
            sim_mat = cosine_similarity(tfidf_matrix)
            sim_mat[sim_mat < threshold] = 0
            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            top_indices = sorted(scores, key=scores.get, reverse=True)[:top_n]
            top_indices.sort()
            return "\n\n".join([sentences[i] for i in top_indices])
        except: return "\n\n".join(sentences[:top_n])

    # --- 5. MMR (중복 제거 강조) 기반 요약 ---
    def mmr_summary(self, text, top_n=5, lambda_val=0.5):
        sentences = self.split_sentences(text)
        if len(sentences) <= top_n: return " ".join(sentences)
        sent_nouns = [self.get_nouns(s) for s in sentences]
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sent_nouns)
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            sim_mat = cosine_similarity(tfidf_matrix)
            
            summary_indices = [np.argmax(scores)]
            remaining_indices = [i for i in range(len(sentences)) if i not in summary_indices]
            
            while len(summary_indices) < top_n and remaining_indices:
                mmr_values = []
                for i in remaining_indices:
                    relevance = scores[i]
                    redundancy = max([sim_mat[i][j] for j in summary_indices])
                    mmr_val = lambda_val * relevance - (1 - lambda_val) * redundancy
                    mmr_values.append((mmr_val, i))
                
                if not mmr_values: break
                next_idx = max(mmr_values, key=lambda x: x[0])[1]
                summary_indices.append(next_idx)
                remaining_indices.remove(next_idx)
            
            summary_indices.sort()
            return "\n\n".join([sentences[i] for i in summary_indices])
        except: return "\n\n".join(sentences[:top_n])

    # --- 6. Ollama AI 요약 ---
    def ollama_summary(self, text):
        prompt = f"""다음 기사를 정확히 5개의 문장으로 요약하라.

                    [출력 규칙]
                    - 반드시 5줄만 출력한다.
                    - 각 줄은 하나의 완전한 문장만 포함한다.
                    - 줄 구분은 단일 개행 문자(\n)만 사용한다.
                    - 줄 사이에 빈 줄을 절대 넣지 않는다.
                    - 번호, 기호, 불릿, 설명을 절대 포함하지 않는다.
                    - 첫 줄 앞과 마지막 줄 뒤에 공백이나 개행을 넣지 않는다.
                    - 출력은 오직 요약문 5줄만 작성한다.

                    {text}"""
        try:
            response = ollama.chat(model=self.ollama_model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            content = response['message']['content']
            
            # 후처리: 문장 끝(., ?, !) 뒤에 오는 공백을 줄바꿈 2개로 변경
            formatted_content = re.sub(r'(?<=[.!?])\s+', '\n\n', content)
            return formatted_content
        except Exception as e:
            return f"Ollama Error: {str(e)}"
