# Runtime GeoJSON maps

Place floor GeoJSON files here. Runtime map matching reads features with a floor/level and
a navigable kind such as room, corridor, walkway, stairs, lift, entrance, or lobby.

Set `properties.navigable=false` for geometry that must never receive a snapped user position.
A* routing remains separate and can use the same GeoJSON graph after localization.
