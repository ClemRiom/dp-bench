"""Add an algorithm: write, paste or upload the code of a mechanism, check it, then save it in the workspace."""

from __future__ import annotations

import hashlib
import re
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from dpbench import workspace
from dpbench.classic.external import LANGUAGES, check_program
from dpbench.plugin_check import check_plugin_code
from dpbench.templates import (
    ACCOUNTANT_TEMPLATE,
    DATASET_TEMPLATE,
    FUNCTION_TEMPLATE,
    QUERY_TEMPLATE,
    TRAINING_TEMPLATE,
)

KINDS = {
    "training": ("Training algorithm (machine learning, Python)", TRAINING_TEMPLATE),
    "function": ("Query mechanism as a function (Python)", FUNCTION_TEMPLATE),
    "query": ("Query mechanism as a class (Python)", QUERY_TEMPLATE),
    "external": ("Query mechanism in another language", ""),
    "dataset": ("Dataset loader (Python)", DATASET_TEMPLATE),
    "accountant": ("Privacy accountant (Python)", ACCOUNTANT_TEMPLATE),
}
WHERE = {
    "mechanism": "the **Run an experiment** page (machine learning)",
    "query_mechanism": "the **Classic DP** page",
    "dataset": "the **Run an experiment** page (datasets)",
    "table": "the **Classic DP** page (tables)",
    "accountant": "the **Run an experiment** and **Accountants** pages",
}
MONOSPACE = "<style>textarea { font-family: ui-monospace, 'Cascadia Code', Menlo, Consolas, monospace !important; font-size: 13px !important; }</style>"


def registered_name(code: str) -> str | None:
    """The name given to the first ``@register_...("name")`` decorator of a plugin's code."""
    match = re.search(r"@(?:register_\w+|noise_mechanism)\(\s*[\"']([A-Za-z_]\w*)[\"']", code or "")
    return match.group(1) if match else None


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _code_input(key: str, template: str, extension: str) -> str | None:
    source = st.segmented_control(
        "Code", ["Write or paste", f"Upload a {extension} file"], default="Write or paste", key=f"source_{key}"
    )
    if source and source.startswith("Upload"):
        uploaded = st.file_uploader(f"File ({extension})", type=[extension.lstrip(".")], key=f"upload_{key}")
        if uploaded is None:
            return None
        code = uploaded.getvalue().decode("utf-8", errors="replace")
        st.code(code, language="python" if extension == ".py" else None, line_numbers=True)
        return code
    return st.text_area("Code (edit it, or replace it by yours)", value=template, height=560, key=f"code_{key}")


def _show_python_report(report: dict[str, Any]) -> None:
    if report.get("error"):
        st.error(report["error"])
        if report.get("traceback"):
            st.code(report["traceback"], language=None)
    for entry in report.get("plugins", []):
        title = f"{entry['kind']} `{entry['name']}`"
        if entry.get("ok"):
            st.success(
                f"{title}: test passed in {entry['seconds']:.1f} s — {entry.get('test', '')}",
            )
            details = {k: v for k, v in entry.items() if k not in ("kind", "name", "ok", "seconds", "test", "summary")}
            if details:
                st.json(details, expanded=False)
        else:
            st.error(f"{title}: {entry.get('error', 'failed')}")
            if entry.get("traceback"):
                st.code(entry["traceback"], language=None)


def _python(kind: str) -> None:
    label, template = KINDS[kind]
    code = _code_input(kind, template, ".py")
    if not code:
        return
    name = registered_name(code) or "my_plugin"
    filename = st.text_input("File name in the workspace", f"{name}.py", key=f"filename_{kind}")
    digest = _digest(code, filename)
    left, right = st.columns(2)
    if left.button("Check the code", key=f"check_{kind}", width="stretch"):
        with st.spinner("Importing the file and running a short test in a separate process..."):
            st.session_state[f"report_{kind}"] = {
                "digest": digest,
                "report": check_plugin_code(code, Path(filename).name),
            }
    state = st.session_state.get(f"report_{kind}")
    checked = state is not None and state["digest"] == digest
    if checked:
        _show_python_report(state["report"])
    elif state is not None:
        st.info("The code changed since the last check: check it again before saving.")
    ok = checked and state["report"].get("ok")
    if right.button(
        "Save to the workspace",
        type="primary",
        disabled=not ok,
        key=f"save_{kind}",
        width="stretch",
    ):
        try:
            result = workspace.save_plugin(filename, code)
        except Exception as error:
            st.error(f"Not saved: {type(error).__name__}: {error}")
            return
        st.cache_data.clear()
        places = [WHERE.get(p.split(":")[0], p) for p in result["plugins"]]
        st.success(
            f"Saved as `{result['path']}`. Registered: {', '.join(f'`{p}`' for p in result['plugins'])}. "
            f"Use it from {', '.join(dict.fromkeys(places))}.",
        )


def _external() -> None:
    def option(language: str) -> str:
        info = LANGUAGES[language]
        return f"{info.title} — {'installed' if info.available else 'not found on this computer'}"

    language = st.selectbox("Language", list(LANGUAGES), format_func=option, key="external_language")
    info = LANGUAGES[language]
    st.markdown(
        "DP-Bench runs the program for every release, with this protocol (any language can follow it):\n\n"
        f"- command: `{' '.join(info.command).replace('{file}', 'PROGRAM')} EPSILON DELTA L1_SENSITIVITY "
        "L2_SENSITIVITY SEED`\n"
        "- **standard input**: the exact answer, one number per line;\n"
        "- **standard output**: the noisy answer, one number per line.\n\n"
        "It can answer every numeric query (counts, sums, means, histograms, centroids, range queries answered one by "
        "one). Training algorithms stay in Python: they need the model and its gradients inside the Python process."
    )
    if not info.available:
        st.warning(f"`{info.command[0]}` is not installed or not on the PATH: the program cannot run on this computer.")
    code = _code_input(f"external_{language}", info.template, info.extension)
    if not code:
        return
    c1, c2, c3 = st.columns([2, 3, 2])
    name = c1.text_input("Mechanism name", f"laplace_{language}", key=f"external_name_{language}")
    description = c2.text_input(
        "Description", f"Laplace mechanism written in {info.title}", key=f"external_desc_{language}"
    )
    uses_delta = c3.checkbox("(ε, δ)-DP (uses δ)", value=False, key=f"external_delta_{language}")
    digest = _digest(code, name, language)
    left, right = st.columns(2)
    if left.button("Check the program", key="check_external", width="stretch"):
        with st.spinner("Running the program on test inputs..."):
            directory = Path(tempfile.mkdtemp(prefix="dpbench-external-"))
            program = directory / f"program{info.extension}"
            program.write_text(code, encoding="utf-8")
            report = check_program(language, program)
            program.unlink(missing_ok=True)
            directory.rmdir()
        st.session_state["report_external"] = {"digest": digest, "report": report}
    state = st.session_state.get("report_external")
    checked = state is not None and state["digest"] == digest
    if checked:
        report = state["report"]
        if report.get("ok"):
            st.success(
                f"The program follows the protocol ({report['seconds_per_call']:.2f} s per call). On 2,000 copies of "
                f"the value 100 (sensitivity 1, ε = 1): noise mean {report['noise_mean']:+.3f}, standard deviation "
                f"{report['noise_std']:.3f} (Laplace: {report['laplace_std']:.3f}).",
            )
            if not report.get("random"):
                st.warning("The output does not change with the seed: is noise added?")
            if not report.get("reproducible"):
                st.info(
                    "The same seed gives different outputs: the runs will not be reproducible (use the seed argument)."
                )
        else:
            st.error(report.get("error", "the check failed"))
    ok = checked and state["report"].get("ok")
    if right.button(
        "Save to the workspace",
        type="primary",
        disabled=not ok,
        key="save_external",
        width="stretch",
    ):
        try:
            manifest = workspace.save_external(name, language, code, description, uses_delta)
        except Exception as error:
            st.error(f"Not saved: {error}")
            return
        st.success(
            f"Saved: query mechanism `{manifest['name']}` ({info.title}). Select it on the **Classic DP** page.",
        )


def _installed() -> None:
    st.subheader("Algorithms in the workspace")
    plugins, externals = workspace.list_plugins(), workspace.list_external()
    if not plugins and not externals:
        st.info("Nothing added yet.")
        return
    for item in plugins:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            registered = ", ".join(f"`{p}`" for p in item["plugins"]) or ":red[nothing registered]"
            left.markdown(
                f"**{item['file']}** (Python) — {registered}"
                + (f"  \n:red[{item['error']}]" if item.get("error") else "")
            )
            if right.button("Delete", key=f"delete_plugin_{item['file']}"):
                workspace.delete_plugin(item["file"])
                st.rerun()
            with left.expander("Code"):
                st.code(Path(item["path"]).read_text(encoding="utf-8"), language="python", line_numbers=True)
    for item in externals:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            title = LANGUAGES.get(item.get("language", ""), None)
            left.markdown(
                f"**{item['name']}** ({title.title if title else item.get('language', '?')}) — query mechanism `{item['name']}`"
            )
            if right.button("Delete", key=f"delete_external_{item['name']}"):
                workspace.delete_external(item["name"])
                st.rerun()
            path = Path(item.get("path", ""))
            if path.is_file():
                with left.expander("Code"):
                    st.code(path.read_text(encoding="utf-8"), language=None, line_numbers=True)


def render() -> None:
    st.html(MONOSPACE)
    st.title("Add an algorithm")
    st.write("The code you add runs on this computer with your rights: only add code you trust.")
    kind = st.radio("What do you want to add?", list(KINDS), format_func=lambda k: KINDS[k][0], horizontal=True)
    if kind == "external":
        _external()
    else:
        _python(kind)
    st.divider()
    _installed()
