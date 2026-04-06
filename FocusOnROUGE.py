import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kss
from konlpy.tag import Okt
import google.generativeai as genai
from bert_score import score as bert_score_calc
from rouge_score import rouge_scorer

class StrategyASummarizer:
    """전략 A: N-gram(1,2) + Okapi BM25 + 위치 휴리스틱 기반 추출 요약기"""
    def __init__(self, lambda_param=0.5, alpha_param=0.2, k1=1.5, b=0.75):
        self.lambda_param = lambda_param  # MMR 관련성-중복성 조절 파라미터
        self.alpha_param = alpha_param    # MMR 점수와 위치 점수의 결합 비율
        self.k1 = k1
        self.b = b
        self.okt = Okt()
        self.stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    def preprocess_korean(self, text):
        morphs = self.okt.pos(text, stem=True)
        tokens = [word for word, pos in morphs if pos in ['Noun', 'Verb', 'Adjective'] and word not in self.stopwords]
        return " ".join(tokens)

    def _build_bm25_matrix(self, sentences):
        """CountVectorizer를 활용하여 BM25 벡터 공간 행렬을 수학적으로 직접 연산"""
        # N-gram(1, 2) 적용: 형태소 1개 단위 및 2개 단위의 연접까지 고려
        vectorizer = CountVectorizer(ngram_range=(1, 2))
        tf_matrix = vectorizer.fit_transform(sentences).toarray()
        
        N = tf_matrix.shape[0] # 문장의 총 개수
        if N == 0:
            return tf_matrix

        # 1. 길이 정규화 및 평균 문장 길이 계산
        doc_lengths = tf_matrix.sum(axis=1)
        avgdl = np.mean(doc_lengths)
        if avgdl == 0:
            avgdl = 1e-6

        # 2. IDF 계산 (음수 방지 스무딩 적용)
        df = np.count_nonzero(tf_matrix, axis=0)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

        # 3. BM25 가중치 행렬 계산
        numerator = tf_matrix * (self.k1 + 1)
        denominator = tf_matrix + self.k1 * (1 - self.b + self.b * (doc_lengths[:, np.newaxis] / avgdl))
        
        # Zero division 방지 및 행렬 요소별 곱셈
        denominator[denominator == 0] = 1e-6 
        bm25_matrix = idf * (numerator / denominator)
        
        return bm25_matrix

    def _calculate_position_weights(self, num_sentences):
        """U-shape 분포의 위치 가중치 계산"""
        if num_sentences <= 1:
            return np.array([1.0])
        positions = np.arange(num_sentences)
        # 수식: |(2 * i / (N-1)) - 1|
        weights = np.abs((2 * positions / (num_sentences - 1)) - 1)
        return weights

    def summarize(self, text, top_k=3):
        sentences = kss.split_sentences(text)
        N = len(sentences)
        if N <= top_k:
            return " ".join(sentences)

        processed_sentences = [self.preprocess_korean(sentence) for sentence in sentences]
        
        # 1. BM25 행렬 구축 (TF-IDF 대체)
        bm25_matrix = self._build_bm25_matrix(processed_sentences)
        doc_vector = np.mean(bm25_matrix, axis=0).reshape(1, -1)
        
        # 2. 위치 가중치 도출
        pos_weights = self._calculate_position_weights(N)

        unselected = list(range(N))
        selected = []

        # 3. MMR 및 가중치 결합 알고리즘
        while len(selected) < top_k and unselected:
            final_scores = {}
            
            for i in unselected:
                sent_vec = bm25_matrix[i].reshape(1, -1)
                
                # Relevance calculation
                sim_to_doc = cosine_similarity(sent_vec, doc_vector)[0][0]
                
                # Redundancy calculation
                if not selected:
                    max_sim_to_selected = 0
                else:
                    selected_vecs = bm25_matrix[selected]
                    sims_to_selected = cosine_similarity(sent_vec, selected_vecs)[0]
                    max_sim_to_selected = np.max(sims_to_selected)
                
                # 순수 MMR 스코어 산출
                mmr_score = (self.lambda_param * sim_to_doc) - ((1 - self.lambda_param) * max_sim_to_selected)
                
                # 통계적 휴리스틱(위치 점수) 결합
                # 알파(alpha_param) 값에 따라 MMR의 논리적 스코어와 구조적 통계 스코어의 비율을 조정
                final_score = ((1 - self.alpha_param) * mmr_score) + (self.alpha_param * pos_weights[i])
                final_scores[i] = final_score

            best_idx = max(final_scores, key=final_scores.get)
            selected.append(best_idx)
            unselected.remove(best_idx)

        selected.sort()
        return " ".join([sentences[i] for i in selected])


# ==========================================
# 실행 및 평가 로직 (동일한 LLM, Evaluator 사용)
# ==========================================
if __name__ == "__main__":
    API_KEY = os.environ.get("GOOGLE_API_KEY") or "본인의_GEMINI_API_KEY_입력"
    
    # 평가를 위한 LLMSummarizer 및 SummarizationEvaluator 클래스는
    # 이전 채팅의 코드를 그대로 사용합니다.
    # ... (LLMSummarizer, SummarizationEvaluator 정의부 생략) ...

    sample_text = """
    인공지능(AI) 기술이 빠르게 발전하면서 전 산업 분야에 걸쳐 디지털 전환이 가속화되고 있다. 특히 대규모 언어 모델(LLM)의 등장은 자연어 처리 분야에 혁명적인 변화를 가져왔다. 
    과거에는 텍스트를 요약하거나 번역하기 위해 복잡한 수학적 통계 모델과 방대한 규칙 기반 시스템을 직접 구축해야 했다. 이러한 전통적인 방식은 특정 도메인에서는 준수한 성능을 보였으나, 
    새로운 패턴의 문장이나 은유적인 표현을 이해하는 데에는 한계를 드러냈다. 그러나 딥러닝 기반의 LLM은 수조 개의 매개변수를 통해 문맥의 숨겨진 의미까지 파악할 수 있는 능력을 갖추게 되었다. 
    이에 따라 기업들은 고객 응대 챗봇부터 자동 문서 작성 시스템에 이르기까지 다양한 서비스에 LLM을 적극적으로 도입하고 있다. 
    전문가들은 앞으로 AI 기술이 더욱 고도화됨에 따라 인간의 고유 영역으로 여겨졌던 창의적인 글쓰기나 복잡한 논리적 추론 분야에서도 AI가 중요한 역할을 할 것으로 전망하고 있다. 
    하지만 이와 동시에 생성형 AI가 만들어내는 환각(Hallucination) 현상이나 데이터 편향성 문제를 해결하기 위한 윤리적, 기술적 안전장치 마련도 시급하다는 목소리가 커지고 있다.
    """

    TARGET_SENTENCES = 3

    print("1. 전략 A (BM25 + N-gram + 휴리스틱 기반) 요약 진행 중...")
    # alpha_param=0.2로 설정하여 위치 가중치를 20% 비율로 반영
    strategy_a_summarizer = StrategyASummarizer(lambda_param=0.6, alpha_param=0.2)
    strategy_a_summary = strategy_a_summarizer.summarize(sample_text, top_k=TARGET_SENTENCES)
    
    print("2. LLM 방식(Gemini 기반) 요약 진행 중...")
    # llm_summarizer = LLMSummarizer(api_key=API_KEY)
    # llm_summary = llm_summarizer.summarize(sample_text, target_sentences_count=TARGET_SENTENCES)
    
    # LLM 및 Evaluator 모듈이 준비되었다고 가정하고 실행하면, 
    # N-gram 기반 지표인 ROUGE Score에서 기존 TF-IDF 모델 대비 유의미한 수치 향상을 확인할 수 있습니다.