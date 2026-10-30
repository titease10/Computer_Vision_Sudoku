import numpy as np

from charts import CHARTS_CSS, counter, donut_html, gauge_html, heatmap_html, histogram_html, model_card_html, stack_html


def test_counter_splits_number_and_suffix():
    assert 'style="--to:1564"' in counter("1564 ms") and 'data-suf=" ms"' in counter("1564 ms")
    assert 'data-suf="%"' in counter("95%")
    assert counter("OpenCV") == "OpenCV"


def test_heatmap_has_81_cells_with_tiers():
    conf = np.ones((9, 9))
    conf[0, 0], conf[0, 1] = 0.7, 0.3
    out = heatmap_html(conf, [[0] * 9 for _ in range(9)])
    assert out.count("style=\"--i:") == 81
    assert 'class="lo"' in out and 'class="bad"' in out and 'class="t3"' in out


def test_histogram_counts_every_cell():
    conf = np.full((9, 9), 0.999)
    svg = histogram_html(conf)
    assert svg.count("class=\"bar ") == 10 and ">81<" in svg


def test_gauge_clamps_value():
    assert "--p:100.0" in gauge_html(3.0, "x")
    assert "--p:0.0" in gauge_html(-1.0, "x")


def test_donut_and_stack_shares():
    d = donut_html([("a", 30, "c0"), ("b", 50, "c1")], "81", "cases")
    assert d.count("calc(var(--k)") == 4  # 2 parts, début et fin de chacune
    assert "37.50%" in d  # 30 / 80
    s = stack_html([("detect", 0.1), ("warp", 0.3), ("rien", 0.0)])
    assert s.count("<i class=\"c") == 4  # 2 segments + 2 pastilles de légende, l'étape à 0 est ignorée


def test_model_card_skips_missing_metrics():
    out = model_card_html("M", "YOLO", [("a", 0.5), ("b", None)])
    assert out.count('class="mrow"') == 1


def test_css_has_no_angle_bracket():
    # st.html supprime tout <style> dont le contenu ressemble à une balise (ex. "<integer>")
    from charts import CHARTS_CSS

    assert "<" not in CHARTS_CSS


def test_repair_banner_lists_changes_and_is_empty_without_repairs():
    from charts import repair_banner_html

    assert repair_banner_html([]) == ""
    out = repair_banner_html([(2, 0, 5, 6), (4, 4, 0, 3)])
    assert 'role="alert"' in out and "2 cells were changed" in out
    assert "row 3, column 1: <b>5</b> → <b>6</b>" in out and "<b>empty</b> → <b>3</b>" in out
    assert "1 cell was changed" in repair_banner_html([(0, 0, 1, 2)])


def test_board_has_nine_blocks_of_nine_cells_and_custom_tooltips():
    from charts import board_html

    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 5
    solution = [[(r * 3 + r // 3 + c) % 9 + 1 for c in range(9)] for r in range(9)]
    out = board_html(grid, solution, {(0, 1)}, {(0, 2)}, pop=True)
    assert out.count('class="blk"') == 9 and out.count('class="cell') == 81
    assert out.count("found") >= 80 and 'class="board pop"' in out and "--end:" in out
    assert "Row 1, column 1: 5 on the paper" in out and "uncertain reading" in out and "corrected using the sudoku rules" in out
    assert " title=" not in out  # infobulles maison (data-tip), pas celles du navigateur


def test_no_native_title_tooltips_in_charts():
    import numpy as np

    from charts import heatmap_html, stack_html

    assert " title=" not in heatmap_html(np.ones((9, 9)), [[0] * 9 for _ in range(9)])
    assert " title=" not in stack_html([("a", 0.1), ("b", 0.2)])
