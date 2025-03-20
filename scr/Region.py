import cartopy.io.shapereader as shpreader
import geopandas
import pycountry
import pycountry_convert as pc
from shapely import union_all

class Region:
    def __init__(self, name, geom, bounds):
        """
            Initializes a Region object with a name, geometry, and geographical bounds.

            Arguments:
                name {str}: Name of the region.
                geom {Geometry}: Shapely geometry object defining the region's shape.
                bounds {tuple}: Boundaries (min_lon, min_lat, max_lon, max_lat) of the region.
        """
        self.name = name  # Store region's name
        self.geometry = geom  # Store region's geometry
        self.min_lon, self.min_lat, self.max_lon, self.max_lat = bounds  # Set boundaries

class Country(Region):
    def __init__(self, country_name):
        """
            Initializes a Country object by finding and setting the corresponding country
            and continent geometries.

            Arguments:
                country_name {str}: Name of the country to initialize.
        """
        try:
            countries = pycountry.countries.search_fuzzy(country_name)  # Search for country in pycountry
            if len(countries) > 1:
                raise KeyError(f"Country name {country_name} is ambiguous.")  # Error if multiple matches
            pycountry_country = countries[0]  # Get the exact match
        except LookupError:
            raise ValueError(f"Country {country_name} not found in pycountry.")  # Error if not found

        filename = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
        df = geopandas.read_file(filename)  # Load shapefile data

        land_geom = df.loc[df['ADMIN'] == pycountry_country.name]['geometry'].values[0]  # Find country's geometry
        super().__init__(pycountry_country.name, land_geom, land_geom.bounds)  # Initialize as Region with bounds

        country_continent_code = pc.country_alpha2_to_continent_code(pycountry_country.alpha_2)  # Get continent code
        continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)  # Convert code to name

        continent_df = df.loc[df['CONTINENT'] == continent_name]  # Select all countries in the continent
        continent_df = continent_df.loc[continent_df['ADMIN'] != 'Russia']  # Exclude Russia if continent is Europe
        continent_geom = union_all(continent_df['geometry'])  # Combine all geometries into one for the continent

        # Set bounds for Europe or use continent geometry bounds
        if continent_name == 'Europe':
            bounds = -30, 25, 42, 73  # Fixed boundaries for Europe
        else:
            bounds = continent_geom.bounds  # Use the full continent bounds

        self.continent = Region(continent_name, continent_geom, bounds)  # Create Region object for the continent
