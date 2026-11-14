import hashlib
import json
from pathlib import Path

import nbformat


def test_notebook_schema_and_python_cells():
    notebook = nbformat.read("flippy_illusions_sdxl_colab.ipynb", as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell.source for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code_cells) == 7
    for index, source in enumerate(code_cells):
        compile(source, f"colab-cell-{index + 1}", "exec")
    setup = code_cells[0]
    assert "adrianIam/Diffusion-Illusions" in setup
    assert "git(\"status\", \"--porcelain\")" in setup
    assert "rm -rf" not in setup
    assert '"--detach", "FETCH_HEAD"' in setup


def test_legacy_computation_cells_preserved():
    notebook = json.loads(Path("flippy_illusions_for_colab.ipynb").read_text())
    # Source cells after setup are unchanged from inspected fork commit 71646f3.
    digest = hashlib.sha256(json.dumps([c["source"] for c in notebook["cells"][3:]], sort_keys=True).encode()).hexdigest()
    assert digest == "e587b8ba676b61aaac52435c170340846264931392065500381bb87981889c2e"
    assert "rm -rf" not in "".join(notebook["cells"][2]["source"])


def test_comparison_has_matched_pairs_and_seeds(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare_flippy", "scripts/compare_flippy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plans = module.prepare(tmp_path, iterations=10)
    assert len(plans) == 12
    for index in range(0, len(plans), 2):
        sd1, sdxl = plans[index][2], plans[index + 1][2]
        assert sd1.prompt_a == sdxl.prompt_a and sd1.prompt_b == sdxl.prompt_b
        assert sd1.seed == sdxl.seed and sd1.iterations == sdxl.iterations
        assert sd1.model_size == 512 and sdxl.model_size == 1024
        assert sd1.view_sampling == sdxl.view_sampling == "random"
    assert not (tmp_path / "trials").exists()  # Preparing configurations starts no generation.
