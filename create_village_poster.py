#!/usr/bin/env python3
"""Village / Rural Map Poster Generator.

Variant of create_map_poster.py optimized for villages and small settlements:
- Adds extra OSM layers (forests, farmland, waterways, railways, cemeteries)
- Skips the top/bottom gradient fade overlay so every detail stays visible
- Reuses themes, fonts and helper functions from create_map_poster.py

Usage mirrors create_map_poster.py:

    uv run ./create_village_poster.py --city "Leskovec nad Moravicí" \
        --country "Czechia" --theme leskovec_nad_moravici \
        -W 8.3 -H 11.7 -d 2500 --display-country "Česko" --credit "LR"
"""
import argparse
import sys

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import osmnx as ox
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

import create_map_poster as cmp
from lat_lon_parser import parse
from create_map_poster import (
    FONTS,
    fetch_features,
    fetch_graph,
    generate_output_filename,
    get_available_themes,
    get_coordinates,
    get_crop_limits,
    get_edge_colors_by_type,
    get_edge_widths_by_type,
    is_latin_script,
    load_fonts,
    load_theme,
)


def _project(layer, g_proj):
    try:
        return ox.projection.project_gdf(layer)
    except Exception:
        return layer.to_crs(g_proj.graph["crs"])


def _plot_polys(layer, ax, g_proj, *, facecolor="none", edgecolor="none",
                zorder, alpha=1.0, linewidth=0):
    if layer is None or layer.empty:
        return
    polys = layer[layer.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if polys.empty:
        return
    polys = _project(polys, g_proj)
    polys.plot(ax=ax, facecolor=facecolor, edgecolor=edgecolor,
               linewidth=linewidth, alpha=alpha, zorder=zorder)


def _plot_lines(layer, ax, g_proj, color, linewidth, zorder):
    if layer is None or layer.empty:
        return
    lines = layer[layer.geometry.type.isin(["LineString", "MultiLineString"])]
    if lines.empty:
        return
    lines = _project(lines, g_proj)
    lines.plot(ax=ax, color=color, linewidth=linewidth, zorder=zorder)


def create_village_poster(
    theme,
    city,
    country,
    point,
    dist,
    output_file,
    output_format,
    width=12,
    height=16,
    country_label=None,
    display_city=None,
    display_country=None,
    fonts=None,
    credit=None,
    outline_only=False,
    transparent_bg=False,
):
    cmp.THEME = theme
    THEME = theme

    display_city = display_city or city
    display_country = display_country or country_label or country

    print(f"Generating village map for {city}, {country}...")

    with tqdm(total=10, desc="Fetching map data", ncols=70) as pbar:
        compensated_dist = dist * (max(height, width) / min(height, width)) / 4

        pbar.set_description("Street network")
        g = fetch_graph(point, compensated_dist)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        pbar.set_description("Water (polygons)")
        water = fetch_features(
            point, compensated_dist,
            tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
            name="water",
        )
        pbar.update(1)

        pbar.set_description("Waterways (lines)")
        waterway_lines = fetch_features(
            point, compensated_dist,
            tags={"waterway": ["river", "stream", "canal", "ditch"]},
            name="waterway_lines",
        )
        pbar.update(1)

        pbar.set_description("Forest")
        forest = fetch_features(
            point, compensated_dist,
            tags={"landuse": "forest", "natural": "wood"},
            name="forest",
        )
        pbar.update(1)

        pbar.set_description("Farmland & meadows")
        farmland = fetch_features(
            point, compensated_dist,
            tags={"landuse": ["farmland", "meadow", "orchard", "vineyard"]},
            name="farmland",
        )
        pbar.update(1)

        pbar.set_description("Parks & green")
        parks = fetch_features(
            point, compensated_dist,
            tags={"leisure": "park", "landuse": "grass"},
            name="parks",
        )
        pbar.update(1)

        pbar.set_description("Cemeteries & rail")
        cemetery = fetch_features(
            point, compensated_dist,
            tags={"landuse": "cemetery", "amenity": "grave_yard"},
            name="cemetery",
        )
        railway = fetch_features(
            point, compensated_dist,
            tags={"railway": ["rail", "light_rail", "narrow_gauge", "tram"]},
            name="railway",
        )
        pbar.update(1)

        pbar.set_description("Buildings")
        buildings = fetch_features(
            point, compensated_dist,
            tags={"building": True},
            name="buildings",
        )
        pbar.update(1)

        pbar.set_description("Paths & tracks")
        paths = fetch_features(
            point, compensated_dist,
            tags={"highway": ["path", "footway", "track", "cycleway", "pedestrian", "bridleway"]},
            name="paths",
        )
        pbar.update(1)

        pbar.set_description("Residential area")
        residential = fetch_features(
            point, compensated_dist,
            tags={"landuse": "residential"},
            name="residential",
        )
        pbar.update(1)

    print("✓ All data retrieved successfully!")

    print("Rendering village map...")
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    g_proj = ox.project_graph(g)

    farmland_color = THEME.get("farmland", THEME.get("parks", THEME["bg"]))
    forest_color = THEME.get("forest", THEME.get("parks", THEME["bg"]))
    waterway_color = THEME.get("waterway", THEME["water"])
    cemetery_color = THEME.get("cemetery", THEME.get("buildings", THEME["road_residential"]))
    railway_color = THEME.get("railway", THEME["road_motorway"])
    building_color = THEME.get("buildings", THEME["road_residential"])
    residential_color = THEME.get("residential", THEME["road_residential"])

    # Line-art rendering: every area is a stroke, hierarchy via linewidth.
    # Residential boundary intentionally omitted — clipped through buildings.
    _plot_polys(farmland, ax, g_proj, edgecolor=farmland_color,
                linewidth=0.4, zorder=0.3)
    if outline_only:
        _plot_polys(water, ax, g_proj, edgecolor=THEME["water"],
                    linewidth=1.4, zorder=0.5)
    else:
        _plot_polys(water, ax, g_proj, facecolor=THEME["water"],
                    edgecolor="none", linewidth=0, zorder=0.5)
    _plot_lines(waterway_lines, ax, g_proj, waterway_color, linewidth=1.0, zorder=0.55)
    _plot_polys(forest, ax, g_proj, edgecolor=forest_color,
                linewidth=0.6, zorder=0.7)
    _plot_polys(parks, ax, g_proj, edgecolor=THEME["parks"],
                linewidth=0.6, zorder=0.8)
    _plot_polys(cemetery, ax, g_proj, edgecolor=cemetery_color,
                linewidth=0.8, zorder=0.85)
    _plot_polys(buildings, ax, g_proj, edgecolor=building_color,
                linewidth=0.45, zorder=0.9)

    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(g_proj)
    edge_widths = [w * 2.4 for w in get_edge_widths_by_type(g_proj)]

    crop_xlim, crop_ylim = get_crop_limits(g_proj, point, fig, compensated_dist)
    ox.plot_graph(
        g_proj, ax=ax, bgcolor=THEME["bg"],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False, close=False,
    )

    path_color = THEME.get("paths", THEME.get("road_residential", "#888888"))
    _plot_lines(paths, ax, g_proj, path_color, linewidth=1.2, zorder=2.4)

    _plot_lines(railway, ax, g_proj, railway_color, linewidth=1.6, zorder=2.6)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)

    scale_factor = min(height, width) / 12.0
    base_main, base_sub, base_coords, base_attr = 60, 22, 14, 8

    active_fonts = fonts or FONTS
    if active_fonts:
        font_sub = FontProperties(fname=active_fonts["light"], size=base_sub * scale_factor)
        font_coords = FontProperties(fname=active_fonts["regular"], size=base_coords * scale_factor)
    else:
        font_sub = FontProperties(family="monospace", weight="normal", size=base_sub * scale_factor)
        font_coords = FontProperties(family="monospace", size=base_coords * scale_factor)

    if is_latin_script(display_city):
        spaced_city = "  ".join(list(display_city.upper()))
    else:
        spaced_city = display_city

    base_adjusted_main = base_main * scale_factor
    if len(display_city) > 10:
        adjusted_font_size = max(base_adjusted_main * (10 / len(display_city)), 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(fname=active_fonts["bold"], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family="monospace", weight="bold", size=adjusted_font_size)

    text_stroke_color = THEME.get("text_stroke", THEME["bg"])
    text_stroke = [
        path_effects.withStroke(linewidth=1.8, foreground=text_stroke_color),
    ]
    text_stroke_thin = [
        path_effects.withStroke(linewidth=1.0, foreground=text_stroke_color),
    ]

    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=THEME["text"], ha="center", fontproperties=font_main_adjusted,
            zorder=11, path_effects=text_stroke)
    ax.text(0.5, 0.10, display_country.upper(), transform=ax.transAxes,
            color=THEME["text"], ha="center", fontproperties=font_sub,
            zorder=11, path_effects=text_stroke_thin)

    lat, lon = point
    coords_text = (
        f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0
        else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    )
    if lon < 0:
        coords_text = coords_text.replace("E", "W")

    ax.text(0.5, 0.07, coords_text, transform=ax.transAxes,
            color=THEME["text"], alpha=0.7, ha="center", fontproperties=font_coords,
            zorder=11, path_effects=text_stroke_thin)
    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
            color=THEME["text"], linewidth=1 * scale_factor, zorder=11)

    if FONTS:
        font_attr = FontProperties(fname=FONTS["light"], size=6)
    else:
        font_attr = FontProperties(family="monospace", size=6)

    attribution_text = "© OpenStreetMap contributors (ODbL)"
    if credit:
        attribution_text = f"{attribution_text} · {credit}"
    ax.text(0.5, 0.02, attribution_text, transform=ax.transAxes,
            color=THEME["text"], alpha=0.35, ha="center", va="bottom",
            fontproperties=font_attr, zorder=11)

    print(f"Saving to {output_file}...")
    fmt = output_format.lower()
    save_kwargs = dict(bbox_inches="tight", pad_inches=0.05)
    if transparent_bg:
        save_kwargs["transparent"] = True
        ax.set_facecolor("none")
        fig.patch.set_alpha(0)
    else:
        save_kwargs["facecolor"] = THEME["bg"]
    if fmt == "png":
        save_kwargs["dpi"] = 300
    plt.savefig(output_file, format=fmt, **save_kwargs)
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Village/rural map poster generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--city", "-c", type=str, required=True)
    parser.add_argument("--country", "-C", type=str, required=True)
    parser.add_argument("--theme", "-t", type=str, default="terracotta")
    parser.add_argument("--distance", "-d", type=int, default=2500)
    parser.add_argument("--width", "-W", type=float, default=12)
    parser.add_argument("--height", "-H", type=float, default=16)
    parser.add_argument("--latitude", "-lat", type=str, default=None)
    parser.add_argument("--longitude", "-long", type=str, default=None)
    parser.add_argument("--country-label", type=str, default=None)
    parser.add_argument("--display-city", "-dc", type=str, default=None)
    parser.add_argument("--display-country", "-dC", type=str, default=None)
    parser.add_argument("--font-family", type=str, default=None)
    parser.add_argument("--format", "-f", default="png", choices=["png", "svg", "pdf"])
    parser.add_argument("--credit", "-cr", type=str, default=None)
    parser.add_argument("--outline-only", action="store_true",
                        help="Render water as outline only (full line-art mode)")
    parser.add_argument("--transparent-bg", action="store_true",
                        help="Save with transparent background (for t-shirt prints, sticker layouts)")
    args = parser.parse_args()

    args.width = min(args.width, 20.0)
    args.height = min(args.height, 20.0)

    available_themes = get_available_themes()
    if args.theme not in available_themes:
        print(f"Error: Theme '{args.theme}' not found.")
        print(f"Available themes: {', '.join(available_themes)}")
        sys.exit(1)

    custom_fonts = load_fonts(args.font_family) if args.font_family else None
    if args.font_family and not custom_fonts:
        print(f"⚠ Failed to load '{args.font_family}', falling back to Roboto")

    print("=" * 50)
    print("Village Map Poster Generator")
    print("=" * 50)

    if args.latitude and args.longitude:
        coords = [parse(args.latitude), parse(args.longitude)]
        print(f"✓ Coordinates: {coords[0]}, {coords[1]}")
    else:
        coords = get_coordinates(args.city, args.country)

    theme = load_theme(args.theme)
    output_file = generate_output_filename(args.city, args.theme, args.format)

    create_village_poster(
        theme,
        args.city, args.country, coords, args.distance,
        output_file, args.format, args.width, args.height,
        country_label=args.country_label,
        display_city=args.display_city,
        display_country=args.display_country,
        fonts=custom_fonts,
        credit=args.credit,
        outline_only=args.outline_only,
        transparent_bg=args.transparent_bg,
    )

    print("\n" + "=" * 50)
    print("✓ Village poster generation complete!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
