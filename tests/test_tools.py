import json

from app.tools import TOOLS, get_project, list_projects, run_tool


def test_tool_schemas_are_strict_and_closed():
    for t in TOOLS:
        assert t["strict"] is True
        assert t["input_schema"]["additionalProperties"] is False


def test_list_projects_filters_by_category_and_keyword():
    all_ = list_projects()
    mlops = list_projects(category="mlops")
    sm = list_projects(keyword="sagemaker")
    assert all_["count"] > mlops["count"] >= 3
    assert all(p["category"] == "mlops" for p in mlops["projects"])
    assert sm["count"] >= 3 and all("github.com/Data-Fenix/" in p["url"] for p in sm["projects"])


def test_get_project_fuzzy_match_and_miss():
    hit = get_project("smart car parking")
    assert hit["url"].endswith("/smart-car-parking") and "R2" in hit["summary"]
    miss = get_project("quantum teleporter")
    assert "error" in miss


def test_run_tool_handles_unknown_and_bad_args():
    out, err = run_tool("nope", {})
    assert err and "unknown" in json.loads(out)["error"]
    out, err = run_tool("get_project", {"wrong": "x"})
    assert err and "invalid arguments" in json.loads(out)["error"]
    out, err = run_tool("get_project", {"name": "pedata"})
    assert not err and json.loads(out)["title"] == "pedata"
