# Input data specification

All spatial inputs must use the same internally consistent planar coordinate system and units. The software performs nearest-neighbor matching with `scipy.spatial.KDTree`; it does not reproject coordinates.

## Measured-temperature CSV

Required columns:

| Column | Meaning | Unit |
|---|---|---|
| `X` | planar x coordinate | coordinate-system unit |
| `Y` | planar y coordinate | coordinate-system unit |
| `Depth` | burial depth, positive downward | m |
| `temp` | observed formation temperature | °C |

Optional column: `WellID`. If `WellID` is absent, a deterministic identifier is synthesized from rounded `X` and `Y` coordinates.

For each well, an observation at `Depth = 0 m` is used as the surface temperature `Ts` when available. Otherwise, the default `Ts = 18.0 °C` is used.

Duplicate observations at the same well and depth are merged using a `3.0 °C` tolerance around the median. Observations showing an anomalous downward temperature decrease greater than `3.0 °C` relative to the previous retained depth are removed by the implemented profile-cleaning rule.

## Terrestrial heat-flow file

Whitespace-, comma-, or delimiter-detectable triplets:

```text
X  Y  Q
```

`Q` is terrestrial heat flow in **mW/m²**. Non-numeric rows are excluded and may be exported to a bad-row CSV when an export name is supplied.

## Stratigraphic-interface files

Each interface file contains:

```text
X  Y  Z
```

`Z` is interface burial depth in **m**, positive downward in the intended input convention. Interface files must be supplied in stratigraphic order from shallower to deeper boundaries. The implementation uses absolute depths and enforces non-decreasing interface depth internally before layered conductive integration.

At least two valid interface files are required by the complete GUI workflow.

## Spatial matching

Heat-flow nodes form the reference grid. Measured-temperature samples and stratigraphic-interface values are associated by nearest-neighbor `KDTree` matching. Input coordinate systems therefore must be mutually consistent; the code does not infer datum, projection, or coordinate-zone transformations.
