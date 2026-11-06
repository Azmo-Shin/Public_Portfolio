"""앱 연기 시험: 세 탭이 오류 없이 그려지는지 (streamlit AppTest)."""
import os
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")


def test_app_runs_without_exception():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 3
    assert len(at.metric) >= 8


def test_data_files_present():
    d = os.path.join(os.path.dirname(APP), "data")
    names = os.listdir(d)
    assert "seoul_dong.geojson" in names and any(n.startswith("seoul_dong_pop_") for n in names)
