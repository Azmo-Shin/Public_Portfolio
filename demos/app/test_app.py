"""앱 연기 시험: 데모 3개가 각각 오류 없이 그려지는지 (streamlit AppTest, ?demo=1|2|3)."""
import os
import pytest
from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "streamlit_app.py")


@pytest.mark.parametrize("demo", ["1", "2", "3"])
def test_each_demo_runs(demo):
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["demo"] = demo
    at.run()
    assert not at.exception, at.exception
    assert any("class=\"kpis\"" in m.value for m in at.markdown)


def test_data_files_present():
    d = os.path.join(os.path.dirname(APP), "data")
    names = os.listdir(d)
    assert "seoul_dong.geojson" in names and any(n.startswith("seoul_dong_pop_") for n in names)
