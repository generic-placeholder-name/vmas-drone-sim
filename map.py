import pydeck as pdk
import numpy as np

def generate_qgc_waypoints(geo_coords, filename, altitude=20.0):
    waypoints = []
    waypoints.append("QGC WPL 110")
    for i, (lat, lon) in enumerate(geo_coords):
        line = [
            i,                 # index
            0 if i > 0 else 1, # current (1 for start)
            0,                 # autocontinue
            16,                # MAV_CMD_NAV_WAYPOINT
            0, 0, 0, 0,        # params 1-4 (unused)
            lat,               # latitude
            lon,               # longitude
            altitude,          # altitude
            0                  # frame (global)
        ]
        waypoints.append(" ".join(str(x) for x in line))

    with open(filename, "w") as f:
        f.write("\n".join(waypoints))
    print(f" Saved {filename}")


# Bounding box on map (lat, lon) FYI: This is chris farms
map_bounds = np.array([
    [44.853465, -92.816663],
    [44.855128, -92.816765],
    [44.855102, -92.819639],
    [44.853587, -92.819852]
])

# First tour coordinates
first_tour = np.array([
    [105., 150.], [95., 133.], [95., 95.], [57., 95.], [57., 57.], [95., 57.],
    [95., 19.], [57., 19.], [19., 19.], [19., 57.], [19., 95.], [19., 133.],
    [19., 171.], [57., 171.], [95., 171.], [105., 150.]
])

# Second tour coordinates
second_tour = np.array([
    [125., 150.], [133., 171.], [171., 171.], [209., 171.], [209., 133.],
    [209., 95.], [209., 57.], [209., 19.], [171., 19.], [133., 19.],
    [133., 57.], [133., 95.], [171., 57.], [171., 95.], [171., 133.],
    [133., 133.], [125., 150.]
])

def normalize(pixels, geo_bounds):
    pixel_min = np.min(np.vstack([first_tour, second_tour]), axis=0)
    pixel_max = np.max(np.vstack([first_tour, second_tour]), axis=0)
    geo_min, geo_max = geo_bounds.min(axis=0), geo_bounds.max(axis=0)

    lat = geo_max[0] - (pixels[:, 1] - pixel_min[1]) / (pixel_max[1] - pixel_min[1]) * (geo_max[0] - geo_min[0])
    lon = geo_min[1] + (pixels[:, 0] - pixel_min[0]) / (pixel_max[0] - pixel_min[0]) * (geo_max[1] - geo_min[1])
    return np.column_stack((lat, lon))

# Normalize both tours to geographic coordinates
first_geo = normalize(first_tour, map_bounds)
second_geo = normalize(second_tour, map_bounds)

# Format PyDeck inputs
first_path_data = [{"lat": lat, "lon": lon, "order": i} for i, (lat, lon) in enumerate(first_geo)]
second_path_data = [{"lat": lat, "lon": lon, "order": i} for i, (lat, lon) in enumerate(second_geo)]

first_path_coords = [[p["lon"], p["lat"]] for p in first_path_data]
second_path_coords = [[p["lon"], p["lat"]] for p in second_path_data]

# 2D top-down view
view_state = pdk.ViewState(
    latitude=np.mean(first_geo[:, 0]),
    longitude=np.mean(first_geo[:, 1]),
    zoom=18,
    pitch=0,
    bearing=0,
)

# Define layers
first_path_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": first_path_coords}],
    get_color=[255, 0, 0],
    width_scale=1,
    width_min_pixels=1,
    get_width=1,
)

second_path_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": second_path_coords}],
    get_color=[0, 0, 255],
    width_scale=1,
    width_min_pixels=1,
    get_width=1,
)

first_points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=first_path_data,
    get_position='[lon, lat]',
    get_fill_color='[255, 0, 0, 160]',
    get_radius=3,
)

second_points_layer = pdk.Layer(
    "ScatterplotLayer",
    data=second_path_data,
    get_position='[lon, lat]',
    get_fill_color='[0, 0, 255, 160]',
    get_radius=3,
)

# Satellite imagery style
MAPTILER_KEY = "im85a70Qszr71cqMjv5l"
satellite_map_style = f"https://api.maptiler.com/maps/hybrid/style.json?key={MAPTILER_KEY}"

# Combine and render
deck = pdk.Deck(
    layers=[first_path_layer, first_points_layer, second_path_layer, second_points_layer],
    initial_view_state=view_state,
    map_style=satellite_map_style,
    tooltip={"text": "Waypoint {order}"}
)

deck.to_html("two_drone_paths_satellite_2d.html")


