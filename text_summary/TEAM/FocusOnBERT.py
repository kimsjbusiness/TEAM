import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import kss
from konlpy.tag import Okt

class StrategyBSummarizer:
    """전략 B: TF-IDF(Unigram) + LSA(SVD) 기반 의미론적 추출 요약기"""
    def __init__(self, lambda_param=0.7, n_components=3):
        self.lambda_param = lambda_param
        # n_components(k): 축소할 잠재 의미 차원의 수. 
        # 문서의 길이가 짧을 경우 동적으로 조절되어야 합니다.
        self.n_components = n_components 
        self.okt = Okt()
        self.stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    def preprocess_korean(self, text):
        morphs = self.okt.pos(text, stem=True)
        # N-gram=1(Unigram) 원칙: 형태 단위의 결합을 막고 개별 단어의 문맥적 클러스터링을 유도
        tokens = [word for word, pos in morphs if pos in ['Noun', 'Verb', 'Adjective'] and word not in self.stopwords]
        return " ".join(tokens)

    def _build_lsa_matrix(self, processed_sentences):
        """TF-IDF 행렬 구축 및 Truncated SVD를 통한 잠재 의미 공간 변환"""
        # 1. TF-IDF 벡터화
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences)
        
        n_samples, n_features = tfidf_matrix.shape
        
        # SVD 차원(k) 동적 할당: 특이값 분해는 min(문장 수, 단어 수)보다 작은 k를 요구합니다.
        k = min(self.n_components, n_samples - 1, n_features - 1)
        
        # 문장이나 단어 수가 극단적으로 적어 차원 축소가 불가능한 경우 방어 로직
        if k < 1:
            return tfidf_matrix.toarray()

        # 2. Truncated SVD 적용 (A = U * Sigma * V^T 에서 U * Sigma 산출)
        svd = TruncatedSVD(n_components=k, random_state=42)
        lsa_matrix = svd.fit_transform(tfidf_matrix)
        
        return lsa_matrix

    def summarize(self, text, top_k=3):
        sentences = kss.split_sentences(text)
        if len(sentences) <= top_k:
            return " ".join(sentences)

        processed_sentences = [self.preprocess_korean(sentence) for sentence in sentences]
        
        # 3. LSA를 통해 생성된 k차원의 밀집 벡터(Dense Vector) 획득
        lsa_matrix = self._build_lsa_matrix(processed_sentences)
        
        # 문서 전체의 의미를 대변하는 중심점(Centroid) 계산
        doc_vector = np.mean(lsa_matrix, axis=0).reshape(1, -1)
        
        unselected = list(range(len(sentences)))
        selected = []

        # 4. 잠재 의미 공간에서의 MMR 수행
        while len(selected) < top_k and unselected:
            mmr_scores = {}
            for i in unselected:
                sent_vec = lsa_matrix[i].reshape(1, -1)
                
                # Semantic Relevance 계산
                sim_to_doc = cosine_similarity(sent_vec, doc_vector)[0][0]
                
                # Semantic Redundancy 계산
                if not selected:
                    max_sim_to_selected = 0
                else:
                    selected_vecs = lsa_matrix[selected]
                    sims_to_selected = cosine_similarity(sent_vec, selected_vecs)[0]
                    max_sim_to_selected = np.max(sims_to_selected)
                
                mmr_score = (self.lambda_param * sim_to_doc) - ((1 - self.lambda_param) * max_sim_to_selected)
                mmr_scores[i] = mmr_score

            best_idx = max(mmr_scores, key=mmr_scores.get)
            selected.append(best_idx)
            unselected.remove(best_idx)

        selected.sort()
        return " ".join([sentences[i] for i in selected])


# ==========================================
# 실행 로직
# ==========================================
if __name__ == "__main__":
    sample_text = """
    인공지능(AI) 기술이 빠르게 발전하면서 전 산업 분야에 걸쳐 디지털 전환이 가속화되고 있다. 특히 대규모 언어 모델(LLM)의 등장은 자연어 처리 분야에 혁명적인 변화를 가져왔다. 
    과거에는 텍스트를 요약하거나 번역하기 위해 복잡한 수학적 통계 모델과 방대한 규칙 기반 시스템을 직접 구축해야 했다. 이러한 전통적인 방식은 특정 도메인에서는 준수한 성능을 보였으나, 
    새로운 패턴의 문장이나 은유적인 표현을 이해하는 데에는 한계를 드러냈다. 그러나 딥러닝 기반의 LLM은 수조 개의 매개변수를 통해 문맥의 숨겨진 의미까지 파악할 수 있는 능력을 갖추게 되었다. 
    이에 따라 기업들은 고객 응대 챗봇부터 자동 문서 작성 시스템에 이르기까지 다양한 서비스에 LLM을 적극적으로 도입하고 있다. 
    전문가들은 앞으로 AI 기술이 더욱 고도화됨에 따라 인간의 고유 영역으로 여겨졌던 창의적인 글쓰기나 복잡한 논리적 추론 분야에서도 AI가 중요한 역할을 할 것으로 전망하고 있다. 
    하지만 이와 동시에 생성형 AI가 만들어내는 환각(Hallucination) 현상이나 데이터 편향성 문제를 해결하기 위한 윤리적, 기술적 안전장치 마련도 시급하다는 목소리가 커지고 있다.
    """

    TARGET_SENTENCES = 3

    print("1. 전략 B (TF-IDF + LSA(SVD) 기반 의미 추출) 요약 진행 중...")
    # n_components는 문장 수와 데이터 규모에 따라 최적화가 필요합니다. 
    # 짧은 문단이므로 k=3으로 설정하여 3차원 의미 공간으로 투영합니다.
    strategy_b_summarizer = StrategyBSummarizer(lambda_param=0.6, n_components=3)
    strategy_b_summary = strategy_b_summarizer.summarize(sample_text, top_k=TARGET_SENTENCES)
    
    print("\n" + "="*50)
    print(f"[전략 B 요약문 (LSA 적용)]:\n{strategy_b_summary}\n")
    print("="*50 + "\n")
    
    # 평가 모듈(SummarizationEvaluator)에 통과시킬 경우, 
    # 단어의 정확한 형태적 일치를 보는 ROUGE 스코어는 전략 A보다 다소 떨어질 수 있으나,
    # 문맥적 의미 유사도를 보는 BERTScore(Precision/Recall/F1) 측면에서는 방어력 및 향상된 수치를 기대할 수 있습니다.