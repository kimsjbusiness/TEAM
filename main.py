import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import kss
from konlpy.tag import Okt
import google.generativeai as genai
from bert_score import score as bert_score_calc
from rouge_score import rouge_scorer

class TraditionalSummarizer:
    """TF-IDF와 MMR 기반의 전통적 추출 요약 모듈"""
    def __init__(self, lambda_param=0.7):
        self.lambda_param = lambda_param
        self.okt = Okt()
        # 한국어 불용어 사전 (프로젝트 규모에 따라 확장 필요)
        self.stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '한', '하다']

    def preprocess_korean(self, text):
        """형태소 분석기를 통한 명사, 동사, 형용사 추출 및 불용어 제거"""
        morphs = self.okt.pos(text, stem=True)
        tokens = [word for word, pos in morphs if pos in ['Noun', 'Verb', 'Adjective'] and word not in self.stopwords]
        return " ".join(tokens)

    def summarize(self, text, top_k=3):
        # 1. 문장 단위 토큰화
        sentences = kss.split_sentences(text)
        if len(sentences) <= top_k:
            return " ".join(sentences)

        # 2. 전처리 및 TF-IDF 벡터화
        processed_sentences = [self.preprocess_korean(sentence) for sentence in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_sentences).toarray()

        # 문서 전체의 중심점(Centroid) 벡터 계산
        doc_vector = np.mean(tfidf_matrix, axis=0).reshape(1, -1)

        # 3. MMR 알고리즘 적용
        unselected = list(range(len(sentences)))
        selected = []

        while len(selected) < top_k and unselected:
            mmr_scores = {}
            for i in unselected:
                sent_vec = tfidf_matrix[i].reshape(1, -1)
                
                # 관련성 (Relevance to Document)
                sim_to_doc = cosine_similarity(sent_vec, doc_vector)[0][0]
                
                # 중복성 (Redundancy with selected sentences)
                if not selected:
                    max_sim_to_selected = 0
                else:
                    selected_vecs = tfidf_matrix[selected]
                    sims_to_selected = cosine_similarity(sent_vec, selected_vecs)[0]
                    max_sim_to_selected = np.max(sims_to_selected)
                
                # MMR 수식 계산: λ * Sim(Si, D) - (1-λ) * max(Sim(Si, Sj))
                mmr_score = (self.lambda_param * sim_to_doc) - ((1 - self.lambda_param) * max_sim_to_selected)
                mmr_scores[i] = mmr_score

            # MMR 점수가 가장 높은 문장의 인덱스 선택
            best_idx = max(mmr_scores, key=mmr_scores.get)
            selected.append(best_idx)
            unselected.remove(best_idx)

        # 원본 문장의 순서를 유지하여 요약문 생성
        selected.sort()
        return " ".join([sentences[i] for i in selected])


class LLMSummarizer:
    """Gemini API를 활용한 생성 요약 모듈"""
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        # Gemini 1.5 Flash 또는 Pro 모델 지정
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def summarize(self, text, target_sentences_count=5):
        # 통제 변인을 적용한 정교화된 프롬프트
        prompt = f"""다음 제공되는 텍스트를 철저히 분석하여, 전체 내용을 포괄하는 핵심 요약문을 작성하라. 
조건 1: 요약문은 반드시 {target_sentences_count}개의 문장으로만 구성할 것.
조건 2: 원문이 지닌 사실적 정보와 핵심 어휘를 최대한 보존하여 건조한 문체로 작성할 것.

[텍스트]:
{text}
"""
        response = self.model.generate_content(prompt)
        # API 응답 오류 방어 코드
        try:
            return response.text.strip().replace('\n', ' ')
        except Exception as e:
            return f"LLM Generation Error: {e}"


class SummarizationEvaluator:
    """BERTScore 및 ROUGE Score 측정 모듈"""
    @staticmethod
    def evaluate(reference_text, candidate_text):
        # 1. ROUGE Score 측정 (N-gram 기반)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference_text, candidate_text)

        # 2. BERTScore 측정 (문맥적 임베딩 기반)
        # lang="ko"를 통해 한국어 다국어 모델 자동 로드
        P, R, F1 = bert_score_calc([candidate_text], [reference_text], lang="ko", verbose=False)

        return {
            "ROUGE-1 F1": rouge_scores['rouge1'].fmeasure,
            "ROUGE-L F1": rouge_scores['rougeL'].fmeasure,
            "BERTScore Precision": P.item(),
            "BERTScore Recall": R.item(),
            "BERTScore F1": F1.item()
        }


# ==========================================
# 메인 실행 파트 (통합 테스트)
# ==========================================
if __name__ == "__main__":
    # 보안을 위해 API 키는 환경변수로 관리하는 것을 권장
    API_KEY = os.environ.get("GOOGLE_API_KEY") or "본인의_GEMINI_API_KEY_입력"
    
    sample_text = """
    인공지능(AI) 기술이 빠르게 발전하면서 전 산업 분야에 걸쳐 디지털 전환이 가속화되고 있다. 특히 대규모 언어 모델(LLM)의 등장은 자연어 처리 분야에 혁명적인 변화를 가져왔다. 
    과거에는 텍스트를 요약하거나 번역하기 위해 복잡한 수학적 통계 모델과 방대한 규칙 기반 시스템을 직접 구축해야 했다. 이러한 전통적인 방식은 특정 도메인에서는 준수한 성능을 보였으나, 
    새로운 패턴의 문장이나 은유적인 표현을 이해하는 데에는 한계를 드러냈다. 그러나 딥러닝 기반의 LLM은 수조 개의 매개변수를 통해 문맥의 숨겨진 의미까지 파악할 수 있는 능력을 갖추게 되었다. 
    이에 따라 기업들은 고객 응대 챗봇부터 자동 문서 작성 시스템에 이르기까지 다양한 서비스에 LLM을 적극적으로 도입하고 있다. 
    전문가들은 앞으로 AI 기술이 더욱 고도화됨에 따라 인간의 고유 영역으로 여겨졌던 창의적인 글쓰기나 복잡한 논리적 추론 분야에서도 AI가 중요한 역할을 할 것으로 전망하고 있다. 
    하지만 이와 동시에 생성형 AI가 만들어내는 환각(Hallucination) 현상이나 데이터 편향성 문제를 해결하기 위한 윤리적, 기술적 안전장치 마련도 시급하다는 목소리가 커지고 있다.
    """

    TARGET_SENTENCES = 3 # 추출 및 생성할 요약 문장의 수

    print("1. 전통적 방식(MMR 기반) 요약 진행 중...")
    mmr_summarizer = TraditionalSummarizer(lambda_param=0.6) # 다양성 확보를 위해 lambda 0.6 설정
    mmr_summary = mmr_summarizer.summarize(sample_text, top_k=TARGET_SENTENCES)
    
    print("2. LLM 방식(Gemini 기반) 요약 진행 중...")
    llm_summarizer = LLMSummarizer(api_key=API_KEY)
    llm_summary = llm_summarizer.summarize(sample_text, target_sentences_count=TARGET_SENTENCES)

    print("\n" + "="*50)
    print(f"[MMR 요약문]:\n{mmr_summary}\n")
    print(f"[LLM 요약문 (Reference)]:\n{llm_summary}\n")
    print("="*50 + "\n")

    print("3. 요약 품질 평가 진행 중 (LLM 요약문을 기준(Reference)으로 설정)...")
    # 기획안 7번에 따라 LLM 요약문을 Reference로, MMR 요약문을 Candidate로 설정
    eval_results = SummarizationEvaluator.evaluate(reference_text=llm_summary, candidate_text=mmr_summary)

    print("[평가 결과]")
    for metric, score in eval_results.items():
        print(f"- {metric}: {score:.4f}")