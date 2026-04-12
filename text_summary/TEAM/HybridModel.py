import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import kss
from mecab import MeCab

class StrategyCSummarizer:
    """전략 C: [BM25 + N-gram]과 [TF-IDF + LSA]의 하이브리드 앙상블 추출 요약기"""
    def __init__(self, lambda_param=0.7, alpha_param=0.2, beta_param=0.5, k1=1.5, b=0.75, n_components=3):
        # 공통 파라미터
        self.lambda_param = lambda_param
        # 전략 A (Lexical) 파라미터
        self.alpha_param = alpha_param
        self.k1 = k1
        self.b = b
        # 전략 B (Semantic) 파라미터
        self.n_components = n_components
        # 전략 C (Ensemble) 결합 비율 파라미터: 0.5면 두 모델을 1:1로 반영
        self.beta_param = beta_param
        
        self.mecab = MeCab()
        self.stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    def preprocess_korean(self, text):
        morphs = self.mecab.pos(text)
        tokens = [word for word, pos in morphs if pos.startswith(('NN', 'VV', 'VA')) and word not in self.stopwords]
        return " ".join(tokens)

    def _build_bm25_matrix(self, sentences):
        """전략 A: BM25 매트릭스 생성"""
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        tf_matrix = vectorizer.fit_transform(sentences).toarray()
        N = tf_matrix.shape[0]
        if N == 0: return tf_matrix

        doc_lengths = tf_matrix.sum(axis=1)
        avgdl = np.mean(doc_lengths) if np.mean(doc_lengths) > 0 else 1e-6

        df = np.count_nonzero(tf_matrix, axis=0)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

        numerator = tf_matrix * (self.k1 + 1)
        denominator = tf_matrix + self.k1 * (1 - self.b + self.b * (doc_lengths[:, np.newaxis] / avgdl))
        denominator[denominator == 0] = 1e-6 
        
        return idf * (numerator / denominator)

    def _build_lsa_matrix(self, sentences):
        """전략 B: LSA (Truncated SVD) 매트릭스 생성"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        n_samples, n_features = tfidf_matrix.shape
        k = min(self.n_components, n_samples - 1, n_features - 1)
        if k < 1: return tfidf_matrix.toarray()

        svd = TruncatedSVD(n_components=k, random_state=42)
        return svd.fit_transform(tfidf_matrix)

    def _calculate_position_weights(self, num_sentences):
        if num_sentences <= 1: return np.array([1.0])
        positions = np.arange(num_sentences)
        return np.abs((2 * positions / (num_sentences - 1)) - 1)

    def summarize(self, text, top_k=3):
        sentences = kss.split_sentences(text)
        N = len(sentences)
        if N <= top_k:
            return " ".join(sentences)

        processed_sentences = [self.preprocess_korean(sentence) for sentence in sentences]
        
        # --- 두 개의 독립적인 벡터 공간 구축 ---
        # 1. Lexical Space (BM25)
        bm25_matrix = self._build_bm25_matrix(processed_sentences)
        doc_vector_lex = np.mean(bm25_matrix, axis=0).reshape(1, -1)
        pos_weights = self._calculate_position_weights(N)

        # 2. Semantic Space (LSA)
        lsa_matrix = self._build_lsa_matrix(processed_sentences)
        doc_vector_sem = np.mean(lsa_matrix, axis=0).reshape(1, -1)

        unselected = list(range(N))
        selected = []

        # --- 앙상블 MMR 알고리즘 ---
        while len(selected) < top_k and unselected:
            final_scores = {}
            
            for i in unselected:
                # 1) Lexical 점수 연산
                sent_vec_lex = bm25_matrix[i].reshape(1, -1)
                sim_to_doc_lex = cosine_similarity(sent_vec_lex, doc_vector_lex)[0][0]
                if not selected:
                    max_sim_to_selected_lex = 0
                else:
                    selected_vecs_lex = bm25_matrix[selected]
                    max_sim_to_selected_lex = np.max(cosine_similarity(sent_vec_lex, selected_vecs_lex)[0])
                
                mmr_lex = (self.lambda_param * sim_to_doc_lex) - ((1 - self.lambda_param) * max_sim_to_selected_lex)
                score_lex = ((1 - self.alpha_param) * mmr_lex) + (self.alpha_param * pos_weights[i])

                # 2) Semantic 점수 연산
                sent_vec_sem = lsa_matrix[i].reshape(1, -1)
                sim_to_doc_sem = cosine_similarity(sent_vec_sem, doc_vector_sem)[0][0]
                if not selected:
                    max_sim_to_selected_sem = 0
                else:
                    selected_vecs_sem = lsa_matrix[selected]
                    max_sim_to_selected_sem = np.max(cosine_similarity(sent_vec_sem, selected_vecs_sem)[0])
                
                score_sem = (self.lambda_param * sim_to_doc_sem) - ((1 - self.lambda_param) * max_sim_to_selected_sem)

                # 3) 선형 결합 (앙상블)
                final_scores[i] = (self.beta_param * score_lex) + ((1 - self.beta_param) * score_sem)

            # 결합 점수가 가장 높은 문장 추출
            best_idx = max(final_scores, key=final_scores.get)
            selected.append(best_idx)
            unselected.remove(best_idx)

        selected.sort()
        return " ".join([sentences[i] for i in selected])

# ==========================================
# 통합 실행 파트
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

    print("1. 전략 C (Lexical-Semantic 앙상블 기반) 요약 진행 중...")
    # beta_param=0.5 로 설정하여 어휘적 정확도와 의미론적 유사도에 동일한 가중치를 부여합니다.
    strategy_c_summarizer = StrategyCSummarizer(lambda_param=0.6, alpha_param=0.2, beta_param=0.5, n_components=3)
    strategy_c_summary = strategy_c_summarizer.summarize(sample_text, top_k=TARGET_SENTENCES)
    
    print("\n" + "="*50)
    print(f"[전략 C 앙상블 요약문]:\n{strategy_c_summary}\n")
    print("="*50 + "\n")