# Streamlit 앱 배포 가이드

## 방법 1: Streamlit Cloud (가장 쉬움, 추천 ⭐)

### 장점
- ✅ 완전 무료
- ✅ GitHub 연동으로 자동 배포
- ✅ 코드 변경 시 자동 업데이트
- ✅ 설정 간단

### 배포 단계

#### 1. GitHub에 코드 업로드
```bash
# Git 초기화 (아직 안했다면)
git init
git add .
git commit -m "Initial commit: Streamlit timeseries dashboard"
git branch -M main

# GitHub 저장소 생성 후
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

#### 2. requirements.txt 생성
프로젝트 루트에 `requirements.txt` 파일 생성:

```txt
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
numpy>=1.24.0
statsmodels>=0.14.0
scipy>=1.11.0
prophet>=1.1.4
pykalman>=0.9.5
ruptures>=1.1.8
streamlit-lottie>=0.0.5
```

#### 3. Streamlit Cloud에 배포
1. https://share.streamlit.io 접속
2. GitHub 계정으로 로그인
3. "New app" 클릭
4. 저장소 선택
5. Main file path: `3.3 시공간적 불만 트렌드 분석/3.3.1 시계열데이터분석.py`
6. "Deploy!" 클릭

#### 4. 데이터 파일 처리
`df_final.csv` 파일도 GitHub에 업로드해야 합니다.

---

## 방법 2: Heroku

### 배포 단계

#### 1. 필요한 파일 생성

**Procfile** (프로젝트 루트에):
```
web: streamlit run "3.3 시공간적 불만 트렌드 분석/3.3.1 시계열데이터분석.py" --server.port=$PORT --server.address=0.0.0.0
```

**setup.sh** (프로젝트 루트에):
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

**Procfile** 수정:
```
web: sh setup.sh && streamlit run "3.3 시공간적 불만 트렌드 분석/3.3.1 시계열데이터분석.py"
```

#### 2. Heroku 배포
```bash
# Heroku CLI 설치 후
heroku login
heroku create your-app-name
git push heroku main
```

---

## 방법 3: Docker + 클라우드 서비스

### Dockerfile 생성

**Dockerfile** (프로젝트 루트에):
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "3.3 시공간적 불만 트렌드 분석/3.3.1 시계열데이터분석.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 배포
```bash
# Docker 이미지 빌드
docker build -t timeseries-dashboard .

# 로컬 테스트
docker run -p 8501:8501 timeseries-dashboard

# 클라우드에 배포 (AWS ECS, Google Cloud Run, Azure Container Instances 등)
```

---

## 방법 4: 로컬 서버 (내부망용)

### Streamlit 서버 실행
```bash
streamlit run "3.3 시공간적 불만 트렌드 분석/3.3.1 시계열데이터분석.py" --server.port 8501 --server.address 0.0.0.0
```

### Nginx 리버스 프록시 설정 (선택사항)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 방법 5: PythonAnywhere (간단한 호스팅)

1. https://www.pythonanywhere.com 가입
2. Files 탭에서 코드 업로드
3. Web 탭에서 새 웹 앱 생성
4. Streamlit 실행 스크립트 설정

---

## 배포 전 체크리스트

### 필수 파일
- [ ] `requirements.txt` 생성
- [ ] `df_final.csv` 파일 경로 확인
- [ ] 상대 경로 사용 확인

### 코드 수정 필요사항
- [ ] 절대 경로를 상대 경로로 변경
- [ ] 환경 변수 사용 (민감한 정보)
- [ ] 에러 핸들링 강화

### 보안
- [ ] API 키 등 민감한 정보 제거
- [ ] `.streamlit/config.toml` 설정 확인

---

## 추천 순서

1. **Streamlit Cloud** (가장 쉬움, 무료) ⭐
2. **Heroku** (중간 난이도, 무료 티어 있음)
3. **Docker + 클라우드** (고급, 유연함)
4. **로컬 서버** (내부망용)

---

## 문제 해결

### 포트 오류
- `--server.port` 옵션 확인
- 방화벽 설정 확인

### 경로 오류
- 상대 경로 사용 확인
- 파일 존재 여부 확인

### 메모리 부족
- 데이터 크기 최적화
- 캐싱 활용 (`@st.cache_data`)

