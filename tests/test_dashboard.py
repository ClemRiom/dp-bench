"""The dashboard page, run headless with Streamlit's test runner."""

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


def classic_page():
    from dpbench.dashboard.views import classic

    classic.render()


def test_classic_page_runs_a_benchmark():
    app = streamlit_testing.AppTest.from_function(classic_page, default_timeout=300)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    app.slider(key="reps_one").set_value(5)
    app.multiselect(key="eps_one").set_value([0.5, 2.0])
    next(b for b in app.button if b.label == "Run the benchmark").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "ε spent = target ε" and m.value == "yes" for m in app.metric)


def test_classic_page_compares_databases():
    app = streamlit_testing.AppTest.from_function(classic_page, default_timeout=300)
    app.run()
    app.multiselect(key="compare_mechanisms").set_value(["laplace", "exponential"])
    app.multiselect(key="eps_compare").set_value([1.0])
    app.slider(key="reps_compare").set_value(5)
    next(b for b in app.button if b.label == "Compare the databases").click()
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert any(m.label == "Databases × queries" and m.value.startswith("2 ×") for m in app.metric)
