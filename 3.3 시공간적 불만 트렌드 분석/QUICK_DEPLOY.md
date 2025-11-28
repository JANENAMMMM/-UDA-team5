# 빠른 배포 가이드 (Streamlit Cloud)

## 5분 안에 배포하기 🚀

### 1단계: GitHub에 코드 업로드

```bash
# 프로젝트 폴더에서
git init
git add .
git commit -m "Add Streamlit timeseries dashboard"

# GitHub에서 새 저장소 생성 후
git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

**필수 파일 확인:**
- ✅ `requirements.txt` (프로젝트 루트에)
- ✅ `df_final.csv` (데이터 파일)
- ✅ `.streamlit/config.toml` (설정 파일)

### 2단계: Streamlit Cloud 배포

1. **https://share.streamlit.io** 접속
2. **"Sign in"** 클릭 → GitHub 계정으로 로그인
3. **"New app"** 클릭
4. 다음 정보 입력:
   - **Repository**: your-repo-name
   - **Branch**: main
   - **Main file path**: `3.3 시공간적 불만 트렌드 분석/3.3.1 시계열데이터분석.py`
5. **"Deploy!"** 클릭

### 3단계: 완료! 🎉

몇 분 후 앱이 배포됩니다. URL은 다음과 같습니다:
```
https://your-repo-name.streamlit.app
```

---

## 문제 해결

### 파일을 찾을 수 없음
- 파일 경로 확인 (대소문자 구분)
- `df_final.csv` 파일이 저장소에 포함되어 있는지 확인

### 패키지 설치 오류
- `requirements.txt` 확인
- Streamlit Cloud 로그 확인

### 메모리 부족
- 데이터 파일 크기 확인
- 필요시 데이터 샘플링

---

## 배포 후 업데이트

코드를 수정하고 GitHub에 push하면 자동으로 재배포됩니다!

```bash
git add .
git commit -m "Update dashboard"
git push
```

