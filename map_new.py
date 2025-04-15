import re
import numpy as np
import pydeck as pdk
import os

# Bounding box on map (lat, lon) — Chris' farm
# map_bounds = np.array([
#     [44.853465, -92.816663],
#     [44.855128, -92.816765],
#     [44.855102, -92.819639],
#     [44.853587, -92.819852]
# ])

# Updated values attempting to get more distance from tree line
map_bounds = np.array([
    [44.85366044095515, -92.81970832066362],
    [44.85507868903245, -92.8197044208057],
    [44.854965341057, -92.8170447177042],
    [44.85368532280144, -92.8170798164277]
])

# Normalize both tours together
def normalize(pixels, geo_bounds):
    pixel_min = pixels.min(axis=0)
    pixel_max = pixels.max(axis=0)
    geo_min = geo_bounds.min(axis=0)
    geo_max = geo_bounds.max(axis=0)

    lat = geo_max[0] - ((pixels[:, 1] - pixel_min[1]) / (pixel_max[1] - pixel_min[1])) * (geo_max[0] - geo_min[0])
    lon = geo_min[1] + ((pixels[:, 0] - pixel_min[0]) / (pixel_max[0] - pixel_min[0])) * (geo_max[1] - geo_min[1])
    return np.column_stack((lat, lon))

# Parse tour coordinates from .txt file
def parse_tours_from_file(filename):
    with open(filename, "r") as f:
        text = f.read()

    tour1 = re.search(r"first tour:(.*?)second tour:", text, re.DOTALL).group(1)
    tour2 = re.search(r"second tour:(.*?)(first tour costs|second tour costs)", text, re.DOTALL).group(1)

    coords = re.compile(r"tensor\(\[\s*([\d.]+),\s*([\d.]+)\]\)")
    first = np.array([[float(x), float(y)] for x, y in coords.findall(tour1)])
    second = np.array([[float(x), float(y)] for x, y in coords.findall(tour2)])
    return first, second

# Export QGC .waypoints format
def export_qgc_waypoints(geo_coords, filename, altitude=20.0):
    lines = ["QGC WPL 110"]
    for i, (lat, lon) in enumerate(geo_coords):
        lines.append(" ".join(map(str, [
            i, 1 if i == 0 else 0, 0, 16, 0, 0, 0, 0, lat, lon, altitude, 0
        ])))
    with open(filename, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {filename}")

# MAIN
if __name__ == "__main__":
    # Read tours from file
    first_tour, second_tour = parse_tours_from_file("vmas-drone-sim/tours/waypoints/twodrones_mmas_1.txt")
    second_tour[:,1] = second_tour.max(axis=0)[1]-second_tour[:,1]
    first_tour[:,1] = first_tour.max(axis=0)[1]-first_tour[:,1]

    # Normalize together for consistency
    all_pixels = np.vstack([first_tour, second_tour])
    all_geo = normalize(all_pixels, map_bounds)
    first_geo = all_geo[:len(first_tour)]
    second_geo = all_geo[len(first_tour):]

    # Export to .waypoints
    export_qgc_waypoints(first_geo, "drone1_mission.waypoints")
    export_qgc_waypoints(second_geo, "drone2_mission.waypoints")

    # Format for PyDeck
    first_data = [{"lat": lat, "lon": lon, "order": i} for i, (lat, lon) in enumerate(first_geo)]
    second_data = [{"lat": lat, "lon": lon, "order": i} for i, (lat, lon) in enumerate(second_geo)]
    first_path = [[p["lon"], p["lat"]] for p in first_data]
    second_path = [[p["lon"], p["lat"]] for p in second_data]

    # Define 3D view
    view = pdk.ViewState(
        latitude=np.mean(all_geo[:, 0]),
        longitude=np.mean(all_geo[:, 1]),
        zoom=18,
        pitch=60,  # Tilt for 3D
        bearing=0,
    )

    # Use high-quality hybrid satellite map
    MAPTILER_KEY = "im85a70Qszr71cqMjv5l"
    satellite_style = f"https://api.maptiler.com/maps/hybrid/style.json?key={MAPTILER_KEY}"

    # Layers: red (drone 1), blue (drone 2)
    layers = [
        pdk.Layer("PathLayer", data=[{"path": first_path}], get_color=[255, 0, 0], width_scale=1, get_width=1),
        pdk.Layer("ScatterplotLayer", data=first_data, get_position='[lon, lat]', get_fill_color='[255, 0, 0, 160]', get_radius=3),
        pdk.Layer("PathLayer", data=[{"path": second_path}], get_color=[0, 0, 255], width_scale=1, get_width=1),
        pdk.Layer("ScatterplotLayer", data=second_data, get_position='[lon, lat]', get_fill_color='[0, 0, 255, 160]', get_radius=3),
    ]

    # Render the 3D map
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view,
        map_style=satellite_style,
        tooltip={"text": "Waypoint {order}"}
    )

    deck.to_html("two_drone_paths_satellite_3d.html")
    print("Saved: two_drone_paths_satellite_3d.html")
