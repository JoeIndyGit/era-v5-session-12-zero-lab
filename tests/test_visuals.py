from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.visuals import plot_fit_matrix, plot_zero_storyboard


ASSETS = [
    '00_project_cover.svg',
    '01_experiment_architecture.svg',
    '02_state_ownership.svg',
    '03_results_dashboard.svg',
    '01_virtual_cluster.svg',
    '02_collective_flow.svg',
    '04_memory_composition.svg',
    '05_fit_matrix.svg',
    '06_communication_pressure.svg',
    '07_decision_map.svg',
    '08_correctness_trajectory.svg',
]

def test_exported_readme_visuals_exist_and_are_nonempty():
    root = Path(__file__).resolve().parents[1]
    for name in ASSETS:
        path = root / 'assets' / name
        assert path.exists(), name
        assert path.stat().st_size > 1_000, name


def test_visual_functions_render_to_png(tmp_path):
    story = tmp_path / 'story.png'
    fit = tmp_path / 'fit.png'
    fig1 = plot_zero_storyboard(path=story)
    plt.close(fig1)
    fig2 = plot_fit_matrix(path=fit)
    plt.close(fig2)
    assert story.exists() and story.stat().st_size > 10_000
    assert fit.exists() and fit.stat().st_size > 10_000


def test_readme_uses_rendered_flow_diagram_not_mermaid_block():
    root = Path(__file__).resolve().parents[1]
    readme = (root / 'README.md').read_text(encoding='utf-8')
    assert '```mermaid' not in readme
    assert '<img' not in readme.lower()
    assert '<p ' not in readme.lower()
    assert '<table' not in readme.lower()
    assert './assets/01_experiment_architecture.svg' in readme
    for name in ASSETS:
        assert f'./assets/{name}' in readme
