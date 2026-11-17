from PIL import Image
import math
import geopandas as gpd
from shapely.geometry import box

from rasterio.features import rasterize
from rasterio.transform import from_bounds
import pandas as pd

from os import mkdir, path, getcwd

def get_subdirectory(y,z):
    dir = path.join(getcwd(), f'tiles/{z}/{y}/')
    if not path.isdir(dir):
        mkdir(dir)
    return dir

def checkered_pattern(W=120, H=120, checker_size=1):
    pattern = Image.new("1", (W, H), 0)
    pixels = pattern.load()

    for y in range(H):
        for x in range(W):
            if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                pixels[x, y] = 1

    return pattern

ROAD_WIDTHS = {
    "motorway": 24.0,
    "primary":  20.0,
    "secondary": 16.0,
    "tertiary": 12.0,
    "residential": 8.0,
    "cycleway": 3,
}

def tile_bbox(x, y, z):
    """Returns the bounding box of a tile in lon/lat

    This function converts tile coordinates (x, y) at zoom level z
    to a bounding box in longitude and latitude. It divides latitude and
    longitude ranges into 2^z tiles, where z is the zoom level.
    
    Parameters
    ----------
    x : int
        Tile x coordinate
    y : int
        Tile y coordinate
    z : int
        Tile zoom level 

    Returns
    -------
    lon_min : float
        Minimum longitude of the tile
    lat_min : float
        Minimum latitude of the tile
    lon_max : float
        Maximum longitude of the tile
    lat_max : float
        Maximum latitude of the tile
    """
    n = 2 ** z
    lon_min = x / n * 360 - 180
    lon_max = (x + 1) / n * 360 - 180
    lat_min = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat_max = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))    
  
    return lon_min, lat_min, lon_max, lat_max

def tile_from_lonlat(lon, lat, z):
    """Returns the tile coordinates for a given lon/lat at a specific zoom
    level.
    
    This function converts longitude and latitude to tile coordinates (x, y)
    at zoom level z. It divides latitude and longitude ranges into 2^z tiles,
    where z is the zoom level.

    Parameters
    ----------
    lon : float
        Longitude
    lat : float
        Latitude
    z : int
        Zoom level

    Returns
    -------
    x : int
        Tile x coordinate
    y : int
        Tile y coordinate
    """
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

def clip_to_bbox(
        gdf,
        bbox,
        crs="EPSG:4326",
        ):
    """ Clips a GeoDataFrame to a bounding box.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame to clip.
    bbox : tuple
        The bounding box to clip to, in the form (minx, miny, maxx, maxy).
    crs : str, optional
        The coordinate reference system of the bounding box, by default "EPSG:4326".

    Returns
    -------
    geopandas.GeoDataFrame
        The clipped GeoDataFrame.
    """

    bbox_geom = box(*bbox)
    bbox_gdf = gpd.GeoDataFrame(
        {"geometry": [bbox_geom]},
        crs=crs
    )

    gdf = gpd.clip(gdf, bbox_gdf)

    return gdf

def render_roads_to_tile(
        roads_gdf,
        x,
        y,
        z,
        tile_size=120,
        width_map=None,
        ):
    """ Renders roads to a raster tile.

    Parameters
    ----------
    roads_gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing road geometries.
    x : int
        Tile x coordinate.
    y : int
        Tile y coordinate.
    z : int
        Tile zoom level.
    tile_size_x : int, optional
        Width of the tile in pixels, by default 240.
    tile_size_y : int, optional
        Height of the tile in pixels, by default 120.

    Returns
    -------
    PIL.Image.Image
        The rendered tile as a PIL Image.
    """
    lon_min, lat_min, lon_max, lat_max = tile_bbox(x, y, z)
    bbox = (lon_min, lat_min, lon_max, lat_max)

    roads_clip = clip_to_bbox(roads_gdf, bbox)

    if width_map is not None:
        roads_clip = buffer_roads_by_class(roads_clip, width_map)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, tile_size, tile_size)

    shapes = ((geom, 1) for geom in roads_clip.geometry)

    raster = rasterize(
        shapes,
        out_shape=(tile_size, tile_size),
        transform=transform,
        fill=0,
        dtype='uint8'
    )

    img = Image.fromarray(raster * 255).convert("L")

    return img

def buffer_roads_by_class(gdf, width_map):
    """ Buffers roads in a GeoDataFrame by their class.
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing road geometries.
    width_map : dict
        A mapping of road classes to buffer widths in meters.
    
    Returns
    -------
    geopandas.GeoDataFrame
        The buffered GeoDataFrame.
    """

    buffered = []

    # Project to a metric CRS for buffering
    gdf = gdf.to_crs(epsg=3857)

    for road_type, width in width_map.items():
        subset = gdf[gdf["fclass"] == road_type]
        if subset.empty:
            continue

        buf = subset.copy()
        buf["geometry"] = buf.geometry.buffer(width)
        buf["value"] = 1  # raster burn value

        buffered.append(buf[["geometry", "value"]])

    # Combine all buffered geometries
    try:    
        return_gdf = gpd.GeoDataFrame(
            pd.concat(buffered, ignore_index=True),
            crs=gdf.crs
        )
        
    except ValueError:
        # No geometries to buffer
        return_gdf = gpd.GeoDataFrame(
            columns=["geometry", "value"],
            crs=gdf.crs
        )

    return return_gdf.to_crs(epsg=4326)