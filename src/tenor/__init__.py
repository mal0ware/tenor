"""Tenor: factors and forecasts of the US Treasury par yield curve.

Modules
-------
pca            principal components of a yield panel
nelson_siegel  the Nelson-Siegel curve in the Diebold-Li factorisation
forecast       recursive, direct h-step forecasts and the random-walk benchmark
evaluate       forecast-error statistics and the Diebold-Mariano test
data           reading the Treasury's daily par yield files
synthetic      a simulated three-factor panel for tests and smoke runs
published      numbers printed in the source papers, for comparison
pipeline       builds every table and figure
"""

__version__ = "0.1.0"
