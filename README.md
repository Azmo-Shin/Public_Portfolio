<p align="center"><img src="assets/banner.svg" alt="신홍조 · Data Scientist" width="100%"></p>

<p align="center">
<a href="demos/"><img src="https://github.com/Azmo-Shin/Public_Portfolio/actions/workflows/demos.yml/badge.svg" alt="demos"></a>
<img src="https://img.shields.io/badge/text-CC%20BY%204.0-lightgrey" alt="CC BY 4.0">
<img src="https://img.shields.io/badge/code-MIT-blue" alt="MIT">
</p>

<p align="center">
<b>예측 · 공간통계 · 텍스트마이닝 · Gen-AI 에이전트</b>를 실무에 적용해 온 응용 데이터 사이언티스트의 <b>방법론 포트폴리오</b>입니다.<br>
공공기관 분석 PL로 정책 데이터 과제를 수행했고, 이후 외식 프랜차이즈 본사에서 <b>수집 → 데이터마트 → 모델링 → 검증 → 풀스택 BI 배포</b>까지 혼자 구축했습니다.
</p>

> **TL;DR (EN)** — Applied data scientist (7 yrs: public-sector analytics PL → F&B franchise HQ). This repo shows *how* I decide: 11 anonymized case studies (problem → data → process → method → code → validation & limits) plus runnable demos on synthetic data. Company data, brand names and real metrics are intentionally excluded.

> 회사 데이터·브랜드명·점포명·실측 수치·접속 정보는 포함하지 않습니다. 각 폴더는 **문제 설정 → 데이터(종류만) → 프로세스 도식 → 방법 → 코드 → 검증과 한계** 순서로 같고, 화면 캡처는 브랜드명·수치를 가린 것입니다.

## 프로젝트

<table>
<tr>
<td width="50%"><a href="01_sales_forecasting/"><img src="assets/card_01.svg" width="100%" alt="01 시계열 매출 예측"></a></td>
<td width="50%"><a href="02_spatial_esda/"><img src="assets/card_02.svg" width="100%" alt="02 공간통계 ESDA"></a></td>
</tr>
<tr>
<td><a href="03_store_clustering_anomaly/"><img src="assets/card_03.svg" width="100%" alt="03 점포 등급화·이상탐지"></a></td>
<td><a href="04_effect_testing_fdr/"><img src="assets/card_04.svg" width="100%" alt="04 효과 검증"></a></td>
</tr>
<tr>
<td><a href="05_korean_sentiment_kcelectra/"><img src="assets/card_05.svg" width="100%" alt="05 한국어 리뷰 감성"></a></td>
<td><a href="06_genai_tool_agent/"><img src="assets/card_06.svg" width="100%" alt="06 tool-calling 에이전트"></a></td>
</tr>
<tr>
<td><a href="07_rule_based_scoring/"><img src="assets/card_07.svg" width="100%" alt="07 규칙 기반 스코어링"></a></td>
<td><a href="08_personal_agent_video_pipeline/"><img src="assets/card_08.svg" width="100%" alt="08 멀티봇 영상 파이프라인"></a></td>
</tr>
<tr>
<td><a href="09_local_llm_eval_sentiment_diag/"><img src="assets/card_09.svg" width="100%" alt="09 로컬 LLM 평가·감성모델 진단"></a></td>
<td><a href="10_ai_judging_multiagent_site/"><img src="assets/card_10.svg" width="100%" alt="10 멀티에이전트 AI 심사"></a></td>
</tr>
<tr>
<td><a href="11_platform_crowding_regression/"><img src="assets/card_11.svg" width="100%" alt="11 승강장 혼잡도 예측 (공공)"></a></td>
<td><a href="demos/"><img src="assets/card_demos.svg" width="100%" alt="실행 예제 · 글"></a></td>
</tr>
</table>

## 한눈에 보기 — 무엇을 왜 그렇게 했나

<table>
<tr>
<th width="44">#</th>
<th width="190">프로젝트</th>
<th>핵심 판단</th>
</tr>
<tr>
<td width="44" align="center">01</td>
<td width="190"><a href="01_sales_forecasting/">매출 예측</a></td>
<td>튜닝(MSE)과 선택(MAPE) 지표를 나누고, 특일을 뺀 MAPE를 평소 정확도로 봄</td>
</tr>
<tr>
<td width="44" align="center">02</td>
<td width="190"><a href="02_spatial_esda/">공간통계 ESDA</a></td>
<td>KNN 가중치로 이웃 없는 구역을 없애고, 매출 예측이 아닌 <b>스크리닝</b>으로 범위 한정</td>
</tr>
<tr>
<td width="44" align="center">03</td>
<td width="190"><a href="03_store_clustering_anomaly/">점포 등급화·이상탐지</a></td>
<td>전체 평균 밴드는 주말을 전부 이상으로 잡으므로 <b>요일조건부</b> 밴드</td>
</tr>
<tr>
<td width="44" align="center">04</td>
<td width="190"><a href="04_effect_testing_fdr/">효과 검증</a></td>
<td>배정이 무작위가 아니라 <b>인과 대신 연관성</b>으로 한정, 항목 수십 개는 BH-FDR로 보정</td>
</tr>
<tr>
<td width="44" align="center">05</td>
<td width="190"><a href="05_korean_sentiment_kcelectra/">리뷰 감성</a></td>
<td>GPU 없는 서버라 트랜스포머는 <b>룰이 애매한 리뷰에만</b></td>
</tr>
<tr>
<td width="44" align="center">06</td>
<td width="190"><a href="06_genai_tool_agent/">tool-calling 에이전트</a></td>
<td>도구 권한은 모델이 아니라 <b>코드가</b> 정함, 키 없이 도는 테스트</td>
</tr>
<tr>
<td width="44" align="center">07</td>
<td width="190"><a href="07_rule_based_scoring/">규칙 기반 스코어링</a></td>
<td>임베딩보다 <b>"모른다"를 말할 수 있는</b> 규칙 매칭을 선택</td>
</tr>
<tr>
<td width="44" align="center">08</td>
<td width="190"><a href="08_personal_agent_video_pipeline/">멀티봇 영상 파이프라인</a><br><sub>(개인)</sub></td>
<td>작은 모델의 자유 문장을 해석하지 않고 <b>정해진 줄만</b> 읽는 주문서 관문</td>
</tr>
<tr>
<td width="44" align="center">09</td>
<td width="190"><a href="09_local_llm_eval_sentiment_diag/">로컬 LLM 평가·모델 진단</a></td>
<td>벤치마크 대신 <b>업무 과제</b>로 재고, GPU 결과가 한 칸이라도 다르면 전환하지 않음</td>
</tr>
<tr>
<td width="44" align="center">10</td>
<td width="190"><a href="10_ai_judging_multiagent_site/">멀티에이전트 AI 심사</a></td>
<td>플레이하지 않고 추측한 점수는 무효, 조작 문구 방어, <b>최종 판단은 사람</b></td>
</tr>
<tr>
<td width="44" align="center">11</td>
<td width="190"><a href="11_platform_crowding_regression/">승강장 혼잡도 예측</a><br><sub>(공공)</sub></td>
<td>노선 전체 하나가 아니라 <b>역 × 방향마다</b> 모델, 가장 최근 1주를 별도 검증</td>
</tr>
</table>

01~07·09·10은 외식 프랜차이즈 본사 실무, 11은 공공기관 의뢰 과제, 08은 개인 프로젝트입니다.

## 실행해 보기 — [`demos/`](demos/)

폴더의 코드는 핵심 로직을 설명하려고 **간추려 다시 쓴 것**이라 그대로 돌릴 수 없어서, 같은 방법을 **가상 데이터로 새로 만든 예제**를 두었습니다. 테스트는 GitHub Actions에서 자동으로 돕니다.

- [`effect_testing_bh`](demos/effect_testing_bh/) — 진짜 효과 6개를 심은 가상 데이터에서 보정 없는 p<0.05 vs BH-FDR 비교 (04)
- [`forecast_blending`](demos/forecast_blending/) — 피처 누수 차단(`shift(1)`) 테스트, MAPE로 후보 선택, 볼록가중 블렌딩 (01)

## 글 — [`writing/`](writing/)

- [감성 모델이 사람 판정과 어긋날 때 — 원인 가설을 하나씩 지운 기록](writing/01_sentiment_model_diagnosis.md)
- [임베딩 대신 규칙을 고른 이유 — "모른다"를 말할 수 있는 매칭](writing/02_rules_that_say_i_dont_know.md)

## 코드 저장소

- [seoul-living-population-gis](https://github.com/Azmo-Shin/seoul-living-population-gis) — 02의 데이터 기반: 서울 250m 생활인구 적재·공간차원·집계마트·FastAPI·React/deck.gl
- [gemma4-agent-template](https://github.com/Azmo-Shin/gemma4-agent-template) — 06의 전체 템플릿 코드 (오프라인 테스트 CI)
- [meeting-minutes-cli](https://github.com/Azmo-Shin/meeting-minutes-cli) — 회의·강의 음성 → 화자분리 전사 → 로컬 LLM 회의록 CLI (개인)

## 주로 쓰는 도구

Python (pandas · scikit-learn · statsmodels · PyTorch) · SQL · Prophet/XGBoost/LightGBM · GeoPandas/libpysal/QGIS · KcELECTRA/ONNX · Ollama · FastAPI · React/Vue

## 공개 정책

- 회사명·브랜드명·점포명·담당자명·실제 매출·가격·모델 성능 실측치는 싣지 않습니다.
- DB 접속정보·API 키·서비스계정·사내 주소는 코드에서 제거했습니다.
- 회사의 소스 코드는 공개하지 않습니다. 폴더의 코드는 방법을 설명하기 위해 간추려 다시 쓴 것이고, 실행 가능한 예제는 `demos/`(가상 데이터)에 있습니다.
- 라이선스: 글·도식·그림 CC BY 4.0, 코드 MIT ([LICENSE](LICENSE)).

## Contact

- Email: 93hjshin@gmail.com
