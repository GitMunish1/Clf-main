# Optional runtime GeoJSON maps

For the POC, Flutter owns the visible map, POIs and node graph. The backend does not use A*
and does not calculate routes.

This directory is only for optional server-side map matching of a predicted X/Y position.
The app can instead send its already-selected ordered route nodes with the localization packet,
and the location engine will track the nearest/current/next node on that supplied route.

If GeoJSON files are placed here, map matching reads features with a floor/level and a navigable
kind such as room, corridor, walkway, stairs, lift, entrance, or lobby. Set
`properties.navigable=false` for geometry that must never receive a snapped user position.
