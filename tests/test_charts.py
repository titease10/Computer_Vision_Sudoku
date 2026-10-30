import numpy as np

from charts import counter, donut_html, gauge_svg, heatmap_html, histogram_svg, model_card_html, stack_html


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
    svg = histogram_svg(conf)
    assert svg.count("<rect") == 10 and ">81<" in svg


def test_gauge_clamps_value():
    assert "100%" in gauge_svg(3.0, "x") or 'data-suf="%"' in gauge_svg(3.0, "x")
    assert 'stroke-dashoffset:0.00' in gauge_svg(3.0, "x")


def test_donut_and_stack_shares():
    d = donut_html([("a", 30, "c0"), ("b", 50, "c1")], "81", "cases")
    assert d.count('class="seg"') == 2
    s = stack_html([("detect", 0.1), ("warp", 0.3), ("rien", 0.0)])
    assert s.count("<i class=\"c") == 4  # 2 segments + 2 pastilles de légende, l'étape à 0 est ignorée


def test_model_card_skips_missing_metrics():
    out = model_card_html("M", "YOLO", [("a", 0.5), ("b", None)])
    assert out.count('class="mrow"') == 1
