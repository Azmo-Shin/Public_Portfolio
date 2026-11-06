# demos — 가상 데이터로 실제로 돌아가는 예제

**▶ [데모 앱 열기](https://shin-ds-demo.streamlit.app/)** (Streamlit Community Cloud, 설치 없음)

[![demos](https://github.com/Azmo-Shin/Public_Portfolio/actions/workflows/demos.yml/badge.svg)](https://github.com/Azmo-Shin/Public_Portfolio/actions/workflows/demos.yml)

폴더 README의 코드는 방법을 설명하려고 간추린 것이라 그대로는 돌릴 수 없습니다. 여기 예제는 같은 방법을 **가상 데이터로 새로 작성**한 것으로, 누구나 실행하고 테스트할 수 있습니다. 출력되는 수치는 가상 데이터 값이며 실제 업무 성능과 무관합니다.

| 예제 | 보여 주는 판단 | 원래 폴더 |
|---|---|---|
| [`effect_testing_bh/`](effect_testing_bh/) | 항목 30개를 각각 검정하면 우연한 '유의'가 섞인다 → BH-FDR로 거른다. 진짜 효과 6개를 심어 둔 데이터로 보정 전후를 비교 | [04](../04_effect_testing_fdr/) |
| [`forecast_blending/`](forecast_blending/) | 롤링 피처는 `shift(1)` 뒤에(누수 차단), 튜닝은 MSE·선택은 MAPE, 블렌딩 가중치는 검증 구간에서만 | [01](../01_sales_forecasting/) |
| [`app/`](app/) — **Streamlit 데모 앱** | 위 판단을 슬라이더로 직접 바꿔 보기: ① BH-FDR ② 요일 조건부 이상탐지 ③ 서울 생활인구 LISA 핫스팟(**실제 공공데이터**) | [04](../04_effect_testing_fdr/) · [03](../03_store_clustering_anomaly/) · [02](../02_spatial_esda/) |

```bash
pip install -r demos/requirements.txt
cd demos/effect_testing_bh && python effect_test_demo.py
cd ../forecast_blending && python blend_demo.py
cd .. && python -m pytest -q          # 테스트: BH 구현 = scipy, 보정이 가짜 유의를 줄임, 피처 누수 없음, 가중치 볼록
```

### 데모 앱 (`app/`)

```bash
pip install -r demos/app/requirements.txt
streamlit run demos/app/streamlit_app.py
```

- 탭 ①·②는 가상 데이터, 탭 ③은 서울시 공공데이터(행정동 단위 서울 생활인구 2026-07 → 행정동 × 주중/주말 × 시간대 평균으로 줄인 `app/data/`, 경계는 서울시 상권분석서비스 행정동). 출처: 서울특별시 열린데이터광장, **공공누리 1유형(출처표시)**. 다시 만들기: `app/prepare_seoul.py`.
- 앱 연기 시험 `app/test_app.py`(streamlit AppTest)도 GitHub Actions에서 함께 돕니다.

