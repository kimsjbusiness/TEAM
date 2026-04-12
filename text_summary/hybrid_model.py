import pandas as pd
from tqdm import tqdm
from TEAM.HybridModel import StrategyCSummarizer

def main():
    # 1. 파일 불러오기
    input_csv = "aitimes_articles_context.csv"
    print(f"[{input_csv}] 파일을 불러오는 중입니다...")
    df = pd.read_csv(input_csv)
    
    total_count = len(df)
    print(f"총 {total_count}개의 기사를 로드했습니다.")
    
    # 2. 하이브리드 요약기 인스턴스 생성
    print("요약 모델(StrategyCSummarizer)을 초기화하는 중입니다. (초기 로딩 - 약 1~2분 대기)...")
    summarizer = StrategyCSummarizer(lambda_param=0.7, alpha_param=0.2, beta_param=0.5, n_components=3)
    
    # 3. 요약 수행
    TARGET_SENTENCES = 3
    summaries = []
    
    print(f"\n🚀 본격적으로 {total_count}개 데이터에 대한 {TARGET_SENTENCES}줄 요약을 시작합니다!")
    
    # tqdm을 사용하여 터미널에 실시간 진행률 및 예상 남은 시간(ETA) 표시
    for idx, row in tqdm(df.iterrows(), total=total_count, desc="요약 진행률"):
        text = str(row['context'])
        
        # 텍스트가 유효한 경우에만 요약
        if text.strip() and text != "nan":
            try:
                summary = summarizer.summarize(text, top_k=TARGET_SENTENCES)
                summaries.append(summary)
            except Exception as e:
                # 에러 발생 시 프로그램이 멈추지 않고 빈칸을 넣고 넘어감
                summaries.append("") 
        else:
            summaries.append("")
            
    # 4. 데이터프레임 정리 및 저장
    df['hybrid_summary'] = summaries
    
    # 지정하신 대로 'title', 'context', 'hybrid_summary' 3개의 컬럼만 추출
    columns_to_save = ['title', 'context', 'hybrid_summary']
    available_columns = [col for col in columns_to_save if col in df.columns]
    
    final_df = df[available_columns]
    
    output_csv = "hybrid_summary_results_12000.csv"
    final_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 완료 대장정 끝!")
    print(f"최종 결과가 세 줄의 컬럼으로 예쁘게 깎여서 [{output_csv}] 파일에 안전하게 저장되었습니다.")

if __name__ == "__main__":
    main()
