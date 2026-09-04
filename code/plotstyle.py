# -*- coding: utf-8 -*-
"""Plotting setup shared by all figure/analysis scripts.

Replaces the internal runtime helper used during development with a
minimal, dependency-light equivalent: non-interactive backend, seaborn
theme, and unicode-minus fix. Figures scripts call ``setup_plot()`` once
before importing pyplot.
"""


def setup_plot(ctx=None):
    import matplotlib

    matplotlib.use("Agg")
    import seaborn as sns

    sns.set_theme()
    matplotlib.rcParams["axes.unicode_minus"] = False
