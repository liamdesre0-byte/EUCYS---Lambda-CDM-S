"""Publication plotting used by Bayesian validation.

Trace, corner, Hubble diagram, residual, and GetDist exports are those
already implemented in ``bayesian_validation.PublicationPlotter``.
"""

from __future__ import annotations

from .bayesian_validation import (
    PublicationPlotter,
    apply_publication_style,
    export_getdist_plots,
    matplotlib_available,
    plot_confidence_contours,
    plot_corner,
    plot_hubble_diagram,
    plot_parameter_evolution,
    plot_posterior,
    plot_trace,
    render_background_plots,
)

__all__ = [
    "PublicationPlotter",
    "apply_publication_style",
    "export_getdist_plots",
    "matplotlib_available",
    "plot_confidence_contours",
    "plot_corner",
    "plot_hubble_diagram",
    "plot_parameter_evolution",
    "plot_posterior",
    "plot_trace",
    "render_background_plots",
]
