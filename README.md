# Tenor

**Thirty years of Treasury yields, and the three numbers that explain almost all of it.**

> **Status: planned.** This repository holds the design. No results exist yet, and nothing below is a finding. Each box in the roadmap gets ticked when the code and its tests are in.

## The question

The Treasury curve has a yield for every maturity, and they all move every day. How many independent things are actually moving, and can you forecast them?

## What this will contain

- Principal components of daily Treasury par yields: level, slope and curvature
- The share of variance each factor explains, through time
- Nelson-Siegel fits
- Diebold-Li forecasts at 1, 6 and 12 months, against a random walk

## The result it is built to show

Three factors explain about 99% of yield changes. Whether the forecast beat "no change" is the open part.

Whatever the answer turns out to be goes at the top of this file, with one chart, before any installation step.

## Sources

- Litterman and Scheinkman, "Common Factors Affecting Bond Returns", Journal of Fixed Income, 1991
- Diebold and Li, "Forecasting the term structure of government bond yields", Journal of Econometrics 130(2), 2006

Every number that differs from a source will be listed in `PAPER.md` with the likely reason.

## Data

The US Treasury's daily par yield curve. Free, no key, fetched by script.

## How it is tested

The maths is tested before any result is trusted.

- A curve built from three known factors is decomposed back into them
- Nelson-Siegel reproduces its own generated curves
- The random-walk benchmark is computed on exactly the same dates as the forecast

## Layout

```
src/tenor/      the code
tests/             maths tests first
scripts/reproduce.py   one command rebuilds every figure and table
scripts/fetch_data.py  downloads public data into data/ (never committed)
results/           committed figures and small tables
PAPER.md           citations, claims tested, comparison with the sources
```

## Roadmap

- [ ] Project skeleton, CI, and the reproduce command
- [ ] Principal components of daily Treasury par yields: level, slope and curvature
- [ ] The share of variance each factor explains, through time
- [ ] Nelson-Siegel fits
- [ ] Diebold-Li forecasts at 1, 6 and 12 months, against a random walk
- [ ] `PAPER.md` comparison with the sources
- [ ] Results and limits written up at the top of this file

## Limits

- Par yields are themselves a fitted curve. This is a model of a model.
- Beating a random walk on yields is hard, and the original result is known to weaken in later samples. Expect that here.
- This is research code. It is not investment advice and it does not trade.

## The name

Tenor is the time left until a bond pays. The yield curve is one yield per tenor.

## Licence

MIT. See `LICENSE`.
