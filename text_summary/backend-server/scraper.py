import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import re

class BaseExtractor:
    """모든 추출기가 상속받아야 할 기본 클래스"""
    def extract(self, soup):
        raise NotImplementedError("서브 클래스에서 extract 메서드를 구현해야 합니다.")

class AITimesExtractor(BaseExtractor):
    """AI 타임스 전용 추출기"""
    def extract(self, soup):
        # 제목 추출
        title_tag = soup.select_one('h1.heading')
        title = title_tag.get_text(strip=True) if title_tag else "제목을 찾을 수 없음"
        title = title.replace('"', "'")

        # 본문 추출
        content_div = soup.select_one('#article-view-content-div')
        content = ""
        if content_div:
            # 불필요한 스크립트, 스타일, 이미지 캡션(figcaption), TTS 컨테이너 제거
            for tag in content_div(["script", "style", "figcaption", "figure"]):
                tag.decompose()

            # 트위터/X 임베드 제거 (blockquote class="twitter-tweet")
            for tag in content_div.select("blockquote.twitter-tweet"):
                tag.decompose()
            
            # "기사를 읽어드립니다" 버튼 컨테이너 제거
            tts_container = content_div.select_one('#audio-tts-container')
            if tts_container:
                tts_container.decompose()

            # 텍스트 추출 (개행 문자를 공백으로 처리하여 한 줄로 만듦)
            text = content_div.get_text(' ', strip=True)

            # JSON 출력 시 "가 \"로 이스케이프 되는 것을 방지하기 위해 '로 변환
            text = text.replace('"', "'")

            # 1. 사진 설명 제거 (예: (사진=KAIST), [사진=...])
            # 괄호로 시작하고 '사진='이 포함된 패턴 제거
            text = re.sub(r'[\(\[]사진=.*?[\)\]]', '', text)

            # 2. 기자 정보 및 이메일 제거 (문서 끝부분)
            # 텍스트 전체에서 이메일 패턴을 찾아서 그 뒤를 잘라내거나 해당 부분을 제거하는 방식은
            # 한 줄로 합쳐진 상태에서는 위험할 수 있음 (문장 중간에 이메일이 나올 수도 있으므로).
            # 하지만 뉴스 기사 특성상 기자 이메일은 보통 끝에 나옴.
            
            # 이메일 패턴
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            
            # 뒤에서부터 검색하여 기자 이메일이 나오면 그 뒷부분(및 이메일 포함)을 제거하는 전략 시도
            # 혹은 정규식으로 "OOO 기자 email" 패턴을 찾아서 제거
            
            # 간단하게는 이메일이 발견된 위치 이후를 날려버리는 방법이 있지만, 본문에 이메일이 포함된 경우 오작동 위험.
            # 따라서 "기자" + 이메일 패턴을 찾음.
            
            # 이름 + 기자 + 이메일 패턴 (예: 임대준 기자 ydj@aitimes.com)
            # 이름은 2~4글자 가정
            reporter_pattern = r'[가-힣]{2,4}\s*기자\s*' + email_pattern
            
            # 패턴이 매칭되면 해당 부분부터 끝까지 제거하거나, 해당 부분만 제거
            # 보통 끝부분에 위치하므로, 매칭된 부분들을 모두 제거
            text = re.sub(reporter_pattern, '', text).strip()
            
            # 혹시 이메일만 있는 경우 (바이트댄스 등 기업 이메일 제외해야 함에 주의)
            # 맨 뒤에 있는 이메일만 제거하는 것이 안전함.
            
            matches = list(re.finditer(email_pattern, text))
            if matches:
                last_match = matches[-1]
                # 매칭된 이메일이 텍스트의 거의 끝부분(예: 뒤에서 50자 이내)에 있다면 기자 이메일일 확률 높음
                if len(text) - last_match.end() < 50:
                    # 그 앞의 "기자" 단어 확인
                    preceeding_text = text[max(0, last_match.start()-10):last_match.start()]
                    if "기자" in preceeding_text:
                         text = text[:last_match.start()].rstrip()
                         # 앞선 "기자" 단어와 이름도 지워야 깔끔하겠지만, 정규식으로 처리했으면 이미 지워졌을 것.
                         # 만약 정규식(reporter_pattern)으로 안 지워진(이름 형식이 다르거나) 경우를 대비해 이메일만이라도 지움.
            
            # 불필요한 공백 정리 (탭, 연속된 공백 등)
            content = re.sub(r'\s+', ' ', text).strip()
            
        else:
            content = "본문을 찾을 수 없음"

        return {
            "title": title,
            "content": content
        }

class GenericExtractor(BaseExtractor):
    """일반 웹사이트용 폴백 추출기 (휴리스틱 기반)"""
    def extract(self, soup):
        # 제목 추출 시도 (h1 태그 우선, 없으면 title 태그)
        title_tag = soup.find('h1')
        if not title_tag:
            title_tag = soup.find('title')
        title = title_tag.get_text(strip=True) if title_tag else "제목을 찾을 수 없음"

        # 본문 추출 시도 (article 태그 우선)
        content_tag = soup.find('article')
        if content_tag:
             for script in content_tag(["script", "style"]):
                script.decompose()
             content = content_tag.get_text(strip=True)
        else:
            # article 태그가 없으면 p 태그들을 모아서 본문으로 간주
            paragraphs = soup.find_all('p')
            content = '\n'.join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]) # 너무 짧은 문장은 제외
            if not content:
                content = "본문을 추출하지 못했습니다."

        return {
            "title": title,
            "content": content
        }

class ScraperFactory:
    """URL에 따라 적절한 추출기를 반환하는 팩토리 클래스"""
    @staticmethod
    def get_extractor(url):
        domain = urlparse(url).netloc
        
        if 'aitimes.com' in domain:
            return AITimesExtractor()
        else:
            return GenericExtractor()

def scrape_article(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        extractor = ScraperFactory.get_extractor(url)
        return extractor.extract(soup)

    except Exception as e:
        return {"error": str(e)}
