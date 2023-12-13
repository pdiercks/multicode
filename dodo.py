from pathlib import Path


def task_coverage_run():
    source_files = list(Path("./src/multi").glob("*.py"))
    return {
            "file_dep": source_files,
            "actions": ["coverage run --source=src -m pytest tests"],
            "targets": [".coverage"],
            "clean": True,
            }
        

def task_coverage_report():
    return {
            "file_dep": [".coverage"],
            "actions": ["coverage report -m"],
            "uptodate": [False],
            "verbosity": 2,
            }
