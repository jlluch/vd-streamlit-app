import os
import json
import folium
import pandas as pd
import geopandas as gpd
import plotly.express as px
import streamlit as st
import pycountry
from streamlit_folium import st_folium

from utils.config import DATA_PATH, CSV_PUB, CSV_INV, CSV_PRINV, WORLD_MAP

groups = [
    'Europe', 'South America', 'North America', 'Asia',
    'United States'
    ]

@st.cache_data
def load_annual_papers_data():
    """
    Loads and processes the annual scholarly publications data.

    Reads data from the CSV file specified by CSV_PUB, filters out entries
    containing "CSET" in the 'Entity' column, and renames the main data column
    to 'Number of articles'.

    Returns:
        pandas.DataFrame: Processed DataFrame with annual papers data.
                          Returns an empty DataFrame if the source file is not found or is empty.
    """
    try:
        path = os.path.join(DATA_PATH, CSV_PUB)
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Error: The data file for annual papers ({CSV_PUB}) was not found at {path}.")
        return pd.DataFrame()

    if df.empty:
        return df # Return empty df if file was empty

    df = df.query('not Entity.str.contains("CSET")')

    df.rename(columns={
        'Number of articles - Field: All': 'Number of articles'
    }, inplace=True)
    return df


def annual_papers():
    """
    Displays a Streamlit chart for annual scholarly publications.
    Allows users to select an entity to view its specific data in a bar chart,
    or view a scatter plot comparing predefined groups.
    Includes an option for log scale on the Y-axis for the scatter plot.
    """
    df = load_annual_papers_data()
    if df.empty:
        st.warning("No hay datos disponibles sobre publicaciones anuales.")
        return

    options = df['Entity'].unique().tolist()
    # It's unlikely options will be empty if df is not, but good for robustness
    if not options:
        st.warning("No hay entidades disponibles para seleccionar en los datos de publicaciones.")
        return

    entity = st.selectbox(
        'Elige la entidad a visualizar',
        options=options,
        label_visibility='collapsed',
        index=None,
        placeholder='Selecciona un país para inspeccionar a detalle...'
    )

    log_y_axis = False
    if entity is None:
        # Only show log scale checkbox if no specific entity is selected (scatter plot mode)
        log_y_axis = st.checkbox("Usar escala logarítmica para eje Y", value=False)

        fig = px.scatter(
            df.query('Entity == @groups'),
            x='Year',
            y='Number of articles',
            size='Number of articles',
            color='Entity',
            log_y=log_y_axis,
            hover_data={ # Enhanced hover data
                'Entity': True,
                'Year': True,
                'Number of articles': ':,d' # Format number with comma and as integer
            }
        )
        st.plotly_chart(fig, width='stretch') # ensure use_container_width
    else:
        # Bar chart for a single selected entity
        entity_df = df[df['Entity'] == entity]
        if entity_df.empty:
            st.warning(f"No hay datos disponibles para la entidad seleccionada: {entity}.")
            return

        fig = px.bar(entity_df,
                     x='Year',
                     y='Number of articles',
                     title=f'Publicaciones anuales de {entity}')
        st.plotly_chart(fig, width='stretch') # ensure use_container_width

@st.cache_data
def load_global_investment_data():
    """
    Loads and processes the global investment data in generative AI.

    Reads data from the CSV file specified by CSV_INV, renames the investment column
    to 'Investment', and scales its values to billions of USD.

    Returns:
        pandas.DataFrame: Processed DataFrame with global investment data.
                          Returns an empty DataFrame if the source file is not found or is empty.
    """
    try:
        path_csv = os.path.join(DATA_PATH, CSV_INV)
        df = pd.read_csv(path_csv)
    except FileNotFoundError:
        st.error(f"Error: The data file for global investment ({CSV_INV}) was not found at {path_csv}.")
        return pd.DataFrame()

    if df.empty:
        return df

    # Rename column for consistency
    df.rename(columns={
        'Global investment in generative AI': 'Investment'
    }, inplace=True)
    # Scale unit to Billions USD
    df['Investment'] = df['Investment'] / 1e9
    return df

def global_investment():
    """
    Displays a Streamlit line chart for global investment in generative AI
    over time, colored by entity. Also shows key investment statistics for each entity.
    """
    df = load_global_investment_data()
    if df.empty:
        st.warning("No hay datos disponibles sobre inversión global en IA Generativa.")
        return

    st.subheader("Evolución Temporal de la Inversión Mundial")
    fig_line = px.line(
        df, 
        x='Year', 
        y='Investment', 
        color='Entity',
        title='Evolución de la Inversión en mundial en IA Generativa',
        labels={'Investment': 'Inversión (miles de millones USD)', 'Year': 'Año'},
        hover_data={'Investment': ':.2fB'}
    )
    fig_line.update_layout(
        xaxis_title="Año",
        yaxis_title="Inversión Mundial (miles de millones USD)"
    )
    st.plotly_chart(fig_line, width='stretch')

    st.subheader("Estadísticas Clave de Inversión")
    
    unique_entities = df['Entity'].unique()
    num_entities = len(unique_entities)
    cols = st.columns(num_entities if num_entities > 0 else 1)
    
    col_index = 0
    for entity_name in unique_entities:
        entity_df = df[df['Entity'] == entity_name]
        if not entity_df.empty:
            total_investment = entity_df['Investment'].sum()
            peak_investment_row = entity_df.loc[entity_df['Investment'].idxmax()]
            peak_year = peak_investment_row['Year']
            peak_value = peak_investment_row['Investment']

            current_col = cols[col_index % num_entities]
            with current_col:
                st.markdown(f"#### {entity_name}")
                st.metric(
                    label="Inversión Total (Global)",
                    value=f"${total_investment:,.2f}B USD"
                )
                st.metric(
                    label=f"Pico de Inversión ({peak_year})",
                    value=f"${peak_value:,.2f}B USD"
                )
            col_index += 1

@st.cache_data
def load_private_ai_investment_data():
    """
    Loads and processes the total private AI investment data.

    Reads data from the CSV file specified by CSV_PRINV, renames the investment column
    to 'Investment', and scales its values to billions of USD.

    Returns:
        pandas.DataFrame: Processed DataFrame with private AI investment data.
                          Returns an empty DataFrame if the source file is not found or is empty.
    """
    try:
        path_csv = os.path.join(DATA_PATH, CSV_PRINV)
        df = pd.read_csv(path_csv)
    except FileNotFoundError:
        st.error(f"Error: The data file for private AI investment ({CSV_PRINV}) was not found at {path_csv}.")
        return pd.DataFrame()

    if df.empty:
        return df

    # Rename column for consistency
    df.rename(columns={
        'Global total private investment in AI': 'Investment'
    }, inplace=True)
    # Scale unit to Billions USD
    df['Investment'] = df['Investment'] / 1e9
    return df

@st.cache_data
def load_annual_papers_map_data():
    """
    Loads and prepares data for the annual scholarly papers map.

    This involves:
    1. Loading the scholarly papers data (similar to `load_annual_papers_data`).
    2. Converting entity names to ISO A3 country codes using `pycountry`.
       Includes fuzzy matching and some hardcoded fallbacks for common mismatches.
    3. Loading world geographic data (GeoJSON).

    Returns:
        tuple: A tuple containing:
            - papers_df (pandas.DataFrame): DataFrame with papers data, including an 'iso_a3' column.
                                            Returns an empty DataFrame if papers data is not found/empty.
            - world_geo_df (geopandas.GeoDataFrame): GeoDataFrame with world map shapes.
                                                     Returns an empty GeoDataFrame if geo data is not found/empty.
    """
    try:
        papers_path = os.path.join(DATA_PATH, CSV_PUB)
        papers_df = pd.read_csv(papers_path)
    except FileNotFoundError:
        st.error(f"Error: The data file for annual papers ({CSV_PUB}) was not found at {papers_path}.")
        papers_df = pd.DataFrame()

    if papers_df.empty:
        try:
            geo_path = os.path.join(DATA_PATH, WORLD_MAP)
            world_geo_df = gpd.read_file(geo_path)
        except FileNotFoundError:
            st.error(f"Error: The geographic data file ({WORLD_MAP}) was not found at {geo_path}.")
            world_geo_df = gpd.GeoDataFrame()
        return papers_df, world_geo_df


    papers_df = papers_df.query('not Entity.str.contains("CSET")')
    papers_df.rename(columns={
        'Number of articles - Field: All': 'Number of articles'
    }, inplace=True)

    # Add ISO A3 codes to papers_df
    # Initialize column first
    papers_df['iso_a3'] = None
    unique_entities = papers_df['Entity'].unique()

    for entity_name in unique_entities:
        iso_code = None
        try:
            country = pycountry.countries.get(name=entity_name)
            if country:
                iso_code = country.alpha_3
            else:
                results = pycountry.countries.search_fuzzy(entity_name)
                if results:
                    iso_code = results[0].alpha_3
        except LookupError:
            # Try fuzzy search if exact match fails or if it's a common practice
            try:
                results = pycountry.countries.search_fuzzy(entity_name)
                if results:
                    iso_code = results[0].alpha_3
            except Exception:
                print(f"Warning: Fuzzy search failed for entity: {entity_name}")
        except Exception as e:
            print(f"Warning: Could not convert entity '{entity_name}' to ISO A3 code. Error: {e}")

        if iso_code:
            papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = iso_code
        else:
            # Optionally, handle specific known mismatches here if pycountry fails
            if entity_name == "Russia":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "RUS"
            elif entity_name == "Iran": # Common mismatch: "Iran, Islamic Republic of"
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "IRN"
            elif entity_name == "South Korea": # Common mismatch: "Korea, Republic of"
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "KOR"
            elif entity_name == "North Korea":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "PRK"
            elif entity_name == "Vietnam":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "VNM"
            elif entity_name == "Czech Republic": # Now Czechia
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "CZE"
            elif entity_name == "Taiwan":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "TWN"
            elif entity_name == "Moldova":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "MDA"
            elif entity_name == "Bolivia":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "BOL"
            elif entity_name == "Venezuela":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "VEN"
            elif entity_name == "Tanzania":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "TZA"
            elif entity_name == "Syria":
                 papers_df.loc[papers_df['Entity'] == entity_name, 'iso_a3'] = "SYR"
            else:
                 print(f"Warning: Could not convert entity '{entity_name}' to ISO A3 code.")

    try:
        geo_path = os.path.join(DATA_PATH, WORLD_MAP)
        world_geo_df = gpd.read_file(geo_path)
    except FileNotFoundError:
        st.error(f"Error: The geographic data file ({WORLD_MAP}) was not found at {geo_path}.")
        world_geo_df = gpd.GeoDataFrame() 
    return papers_df, world_geo_df

@st.cache_data
def load_annual_investment_map_data():
    """
    Loads and processes data for the annual private AI investment map.

    Reads data from the CSV file specified by CSV_PRINV (private AI investment),
    renames the investment column to 'Investment', and scales its values to
    billions of USD (by dividing by 1e9). Also loads world geographic data.

    Returns:
        tuple: A tuple containing:
            - investment_df (pandas.DataFrame): DataFrame with private AI investment data, scaled to billions.
                                                Returns an empty DataFrame if data is not found/empty.
            - world_geo_df (geopandas.GeoDataFrame): GeoDataFrame with world map shapes.
                                                     Returns an empty GeoDataFrame if geo data is not found/empty.
    """
    try:
        investment_path = os.path.join(DATA_PATH, CSV_PRINV)
        investment_df = pd.read_csv(investment_path)
    except FileNotFoundError:
        st.error(f"Error: The data file for private AI investment ({CSV_PRINV}) was not found at {investment_path}.")
        investment_df = pd.DataFrame()

    if not investment_df.empty:
        investment_df.rename(columns={
            'Global total private investment in AI': 'Investment'
        }, inplace=True)
        investment_df['Investment'] = investment_df['Investment'] / 1e9

    try:
        geo_path = os.path.join(DATA_PATH, WORLD_MAP)
        world_geo_df = gpd.read_file(geo_path)
    except FileNotFoundError:
        st.error(f"Error: The geographic data file ({WORLD_MAP}) was not found at {geo_path}.")
        world_geo_df = gpd.GeoDataFrame()

    return investment_df, world_geo_df


def annual_papers_map_folium():
    """
    Displays a Folium map visualizing annual scholarly publications by country.
    Uses ISO A3 codes for joining publication data with geographic data.
    Includes a slider to select the year and tooltips for interaction.
    """
    df_papers_full, world_geo = load_annual_papers_map_data()

    if df_papers_full is None or df_papers_full.empty: # Defensive check for None as well
        st.warning("Los datos de publicaciones anuales están vacíos o no se pudieron cargar.")
        return
    if world_geo is None or world_geo.empty: # Defensive check for None
        st.warning("Los datos geográficos del mundo están vacíos o no se pudieron cargar.")
        return
    
    # Filtrar solo países (excluir regiones y entries without iso_a3)
    regions_to_exclude = ['Europe', 'South America', 'North America', 'Asia', 'World']
    df_countries = df_papers_full[
        ~df_papers_full['Entity'].isin(regions_to_exclude) &
        df_papers_full['iso_a3'].notna()
    ].copy()
    
    # Selector de año
    if df_countries.empty:
        st.warning("No country data available after ISO conversion and filtering.")
        return
    years = sorted(df_countries['Year'].unique(), reverse=False)
    selected_year = st.slider(
        'Selecciona el año:',
        years[0],
        years[-1],
        years[-1]
    )
    
    # Filtrar y agregar datos por año
    df_year = df_countries[df_countries['Year'] == selected_year].copy()
    df_aggregated = df_year.groupby('iso_a3').agg(
        {'Number of articles': 'sum', 'Entity': 'first'}
    ).reset_index()
    
    # Crear mapa base con configuración mejorada
    m = folium.Map(
        location=[20, 0], 
        zoom_start=2,
        tiles='OpenStreetMap',
        width='100%',  # Usar todo el ancho
        height='600px'  # Altura fija
    )
    
    # Crear el mapa coroplético
    if not df_aggregated.empty:
        geojson_data = world_geo.to_json()
        
        # Crear choropleth
        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name="Publicaciones por país",
            data=df_aggregated,
            columns=['iso_a3', 'Number of articles'],
            key_on="feature.properties.iso_a3",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"Número de publicaciones ({selected_year})",
            nan_fill_color='lightgray',
            nan_fill_opacity=0.3
        )
        choropleth.add_to(m)
        
        data_dict = {}
        for _, row in df_aggregated.iterrows():
            data_dict[row['iso_a3']] = (row['Entity'], row['Number of articles'])

        # Función de estilo para la capa de tooltips (invisible)
        def style_function(feature):
            return {
                'fillColor': 'transparent',
                'color': 'transparent',
                'weight': 0,
                'fillOpacity': 0
            }
        
        # Función para crear tooltips
        def create_tooltip(feature):
            iso_a3_code = feature['properties'].get('iso_a3')
            country_name_display = feature['properties'].get('name', 'País Desconocido')

            tooltip_data = data_dict.get(iso_a3_code)
            
            if tooltip_data:
                original_entity_name, publications = tooltip_data
                display_name_for_tooltip = original_entity_name # Or country_name_display
                tooltip_text = f"""
                <div style='font-family: Arial; font-size: 14px; padding: 8px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
                    <b>{display_name_for_tooltip}</b> ({iso_a3_code})<br>
                    <hr style='margin: 5px 0; border: 1px solid #ddd;'>
                    Publicaciones ({selected_year}): <b style='color: #d73027;'>{int(publications):,}</b>
                </div>
                """
            else:
                tooltip_text = f"""
                <div style='font-family: Arial; font-size: 14px; padding: 8px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
                    <b>{country_name_display}</b> ({iso_a3_code})<br>
                    <hr style='margin: 5px 0; border: 1px solid #ddd;'>
                    <i>Sin datos disponibles</i>
                </div>
                """
            return tooltip_text
        
        # Add custom tooltips using a GeoJson layer
        # This ensures tooltips appear even for countries with no data in df_aggregated
        folium.GeoJson(
            geojson_data,
            style_function=lambda x: {'fillColor':'transparent', 'color':'transparent', 'weight':0},
            tooltip=folium.features.GeoJsonTooltip(
                fields=['name', 'iso_a3'], # Fields from GeoJSON to show in tooltip header
                aliases=['País:', 'ISO A3:'],
                localize=True,
                sticky=True,
                labels=True,
                style="""
                    background-color: white; font-family: sans-serif; font-size: 12px;
                    border: 1px solid black; border-radius: 3px; box-shadow: 3px;
                """,
            )
        ).add_to(m)

        for feature in json.loads(world_geo.to_json())['features']:
            folium.GeoJson(
                feature,
                style_function=lambda x: {'fillColor': 'transparent', 'color': 'transparent', 'weight':0, 'fillOpacity':0},
                tooltip=folium.Tooltip(create_tooltip(feature), sticky=True)
            ).add_to(m)

    # Mostrar estadísticas resumidas
    total_countries_with_data = len(df_aggregated) if not df_aggregated.empty else 0
    total_publications = df_aggregated['Number of articles'].sum() if not df_aggregated.empty else 0
    avg_publications = df_aggregated['Number of articles'].mean() if not df_aggregated.empty else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total países con datos", total_countries_with_data)
    with col2:
        st.metric("Total publicaciones", f"{int(total_publications):,}")
    with col3:
        st.metric("Promedio por país", f"{avg_publications:.1f}")
    
    # Mostrar mapa con configuración mejorada de tamaño
    map_data = st_folium(
        m, 
        width=None,
        height=600,
        returned_objects=["last_object_clicked"],
        key=f"map_{selected_year}"
    )
    
    # Mostrar información del país clickeado
    if map_data['last_object_clicked']: # last_object_clicked now returns the id (iso_a3)
        clicked_iso_a3 = map_data['last_object_clicked']
        
        # Find the corresponding feature in world_geo to get its name for display
        clicked_feature = next((f for f in json.loads(world_geo.to_json())['features'] if f['properties']['iso_a3'] == clicked_iso_a3), None)
        clicked_country_name_display = clicked_feature['properties']['name'] if clicked_feature else clicked_iso_a3

        tooltip_content = data_dict.get(clicked_iso_a3)

        if tooltip_content:
            original_entity_name, publications_value = tooltip_content
            # Use original_entity_name for consistency in display if available, else GeoJSON name
            display_name = original_entity_name if original_entity_name else clicked_country_name_display
            st.success(f"**{display_name} ({clicked_iso_a3})**: {int(publications_value):,} publicaciones en {selected_year}")
        else:
            st.info(f"**{clicked_country_name_display} ({clicked_iso_a3})**: Sin datos disponibles para {selected_year}")
    
    # Tabla con los top 10 países
    # df_aggregated has 'Entity' (original name) and 'Number of articles'
    st.subheader("Top 10 países por publicaciones")
    if not df_aggregated.empty:
        top_countries = df_aggregated.nlargest(10, 'Number of articles')
    else:
        top_countries = pd.DataFrame(columns=['Entity', 'Number of articles', 'iso_a3']) # Empty dataframe
    st.dataframe(
        top_countries,
        column_config={
            "Entity": "País (Nombre Original)", # Clarify it's the original name
            "iso_a3": "ISO A3",
            "Number of articles": st.column_config.NumberColumn(
                "Publicaciones",
                format="%d"
            )
        },
        hide_index=True,
        # Ensure columns are in a sensible order if 'iso_a3' is now included
        column_order=("Entity", "iso_a3", "Number of articles") if 'iso_a3' in top_countries.columns else ("Entity", "Number of articles")
    )

def load_annual_investment_map_data():
    """Loads and processes data for the annual investment map."""
    investment_path = os.path.join(DATA_PATH, CSV_PRINV)
    investment_df = pd.read_csv(investment_path)
    investment_df.rename(columns={
        'Global total private investment in AI': 'Investment'
    }, inplace=True)
    investment_df['Investment'] = investment_df['Investment'] / 1e9 # Corrected to 1e9 for Billions

    try:
        geo_path = os.path.join(DATA_PATH, WORLD_MAP)
        world_geo_df = gpd.read_file(geo_path)
    except FileNotFoundError:
        st.error(f"Error: The geographic data file ({WORLD_MAP}) was not found at {geo_path}.")
        world_geo_df = gpd.GeoDataFrame()

    return investment_df, world_geo_df

def annual_investment_map_folium():
    """
    Displays a Folium map visualizing annual private AI investment by major regions/countries.
    Maps aggregate regional data (e.g., "Europe") to constituent countries on the map.
    Includes a slider to select the year and tooltips for interaction.
    """
    df_investment_full, world_geo = load_annual_investment_map_data()

    if df_investment_full is None or df_investment_full.empty: # Defensive check
        st.warning("Los datos de inversión anual están vacíos o no se pudieron cargar.")
        return
    if world_geo is None or world_geo.empty: # Defensive check
        st.warning("Los datos geográficos del mundo están vacíos o no se pudieron cargar.")
        return
    
    # Filtrar entidades válidas (excluir World para el mapa)
    valid_entities = ['China', 'Europe', 'United States']
    df_filtered = df_investment_full[df_investment_full['Entity'].isin(valid_entities)].copy()
    
    # Selector de año
    # Ensure df_filtered is not empty before trying to access 'Year'
    if df_filtered.empty:
        st.warning("No data available for the selected filters.")
        return
    years = sorted(df_filtered['Year'].unique(), reverse=False)
    selected_year = st.slider(
        'Selecciona el año:',
        years[0],
        years[-1],
        years[-1]
    )
    
    # Filtrar datos por año seleccionado
    df_year = df_filtered[df_filtered['Year'] == selected_year].copy()
    
    # Crear mapa base
    m = folium.Map(
        location=[20, 0], 
        zoom_start=2,
        tiles='OpenStreetMap',
        width='100%',
        height='600px'
    )
    
    # Mapeo de entidades a países/regiones en el GeoJSON
    entity_to_countries = {
        'United States': ['United States of America'],
        'China': ['China'],
        'Europe': [
            'Germany', 'France', 'Italy', 'Spain', 'Poland', 'Romania', 'Netherlands',
            'Belgium', 'Czech Rep.', 'Greece', 'Portugal', 'Sweden', 'Hungary',
            'Austria', 'Belarus', 'Switzerland', 'Bulgaria', 'Serbia', 'Denmark',
            'Finland', 'Slovakia', 'Norway', 'Ireland', 'Croatia', 'Bosnia and Herz.',
            'Albania', 'Lithuania', 'Slovenia', 'Latvia', 'Estonia', 'Macedonia',
            'Moldova', 'Luxembourg', 'Malta', 'Iceland', 'Montenegro', 'Cyprus',
            'United Kingdom', 'Ukraine', 'Czechia'
        ]
    }
    
    # Crear DataFrame expandido para el mapa coroplético
    df_expanded = []
    for _, row in df_year.iterrows():
        entity = row['Entity']
        investment = row['Investment']
        
        if entity in entity_to_countries:
            countries = entity_to_countries[entity]
            for country in countries:
                df_expanded.append({
                    'Entity': country,
                    'Investment': investment,
                    'Original_Entity': entity
                })
    
    df_map = pd.DataFrame(df_expanded)
    
    # Crear el mapa coroplético
    if not df_map.empty:
        geojson_data = world_geo.to_json()
        
        # Crear choropleth
        choropleth = folium.Choropleth(
            geo_data=geojson_data,
            name="Inversión en IA por región",
            data=df_map,
            columns=['Entity', 'Investment'],
            key_on="feature.properties.name",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f"Inversión en IA (miles de millones USD) - {selected_year}",
            nan_fill_color='lightgray',
            nan_fill_opacity=0.3
        )
        choropleth.add_to(m)
        
        # Crear diccionario para tooltips
        data_dict = {}
        original_entity_dict = {}
        for _, row in df_map.iterrows():
            data_dict[row['Entity']] = row['Investment']
            original_entity_dict[row['Entity']] = row['Original_Entity']
        
        # Función para crear tooltips personalizados
        def create_tooltip(feature):
            country_name = feature['properties']['name']
            investment = data_dict.get(country_name, None)
            original_entity = original_entity_dict.get(country_name, None)
            
            if investment is not None:
                tooltip_text = f"""
                <div style='font-family: Arial; font-size: 14px; padding: 8px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
                    <b>{country_name}</b><br>
                    <hr style='margin: 5px 0; border: 1px solid #ddd;'>
                    Región: <b style='color: #2166ac;'>{original_entity}</b><br>
                    Inversión ({selected_year}): <b style='color: #d73027;'>${investment:,.1f}B</b>
                </div>
                """
            else:
                tooltip_text = f"""
                <div style='font-family: Arial; font-size: 14px; padding: 8px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
                    <b>{country_name}</b><br>
                    <hr style='margin: 5px 0; border: 1px solid #ddd;'>
                    <i>Sin datos disponibles</i>
                </div>
                """
            return tooltip_text
        
        # Añadir tooltips personalizados
        for feature in json.loads(geojson_data)['features']:
            country_geojson = folium.GeoJson(
                feature,
                style_function=lambda x: {
                    'fillColor': 'transparent',
                    'color': 'transparent',
                    'weight': 0,
                    'fillOpacity': 0
                },
                tooltip=folium.Tooltip(
                    create_tooltip(feature),
                    sticky=True
                )
            )
            country_geojson.add_to(m)
    
    # Mostrar estadísticas resumidas incluyendo World si está disponible
    df_world = df_investment_full[df_investment_full['Entity'] == 'World']
    if not df_world.empty and selected_year in df_world['Year'].values:
        world_investment = df_world[df_world['Year'] == selected_year]['Investment'].iloc[0]
    else:
        world_investment = df_year['Investment'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Regiones con datos", len(df_year))
    with col2:
        st.metric("Inversión Mundial", f"${world_investment:,.1f}B")
    with col3:
        us_investment = df_year[df_year['Entity'] == 'United States']['Investment'].iloc[0] if len(df_year[df_year['Entity'] == 'United States']) > 0 else 0
        st.metric("Estados Unidos", f"${us_investment:,.1f}B")
    with col4:
        china_investment = df_year[df_year['Entity'] == 'China']['Investment'].iloc[0] if len(df_year[df_year['Entity'] == 'China']) > 0 else 0
        st.metric("China", f"${china_investment:,.1f}B")
    
    # Mostrar mapa
    map_data = st_folium(
        m, 
        width=None,
        height=600,
        returned_objects=["last_object_clicked"],
        key=f"investment_map_{selected_year}"
    )
    
    # Mostrar información del país clickeado
    if map_data['last_object_clicked'] and 'properties' in map_data['last_object_clicked']:
        clicked_country = map_data['last_object_clicked']['properties']['name']
        clicked_investment = data_dict.get(clicked_country, None)
        clicked_region = original_entity_dict.get(clicked_country, None)
        
        if clicked_investment is not None:
            st.success(f"**{clicked_country}** (Región: {clicked_region}): ${clicked_investment:,.1f}B en inversión IA ({selected_year})")
        else:
            st.info(f"**{clicked_country}**: Sin datos disponibles para {selected_year}")
    
    # Gráfico de barras con las regiones
    st.subheader("Inversión por Región")
    if not df_year.empty:
        fig_bar = px.bar(
            df_year, 
            x='Entity', 
            y='Investment',
            title=f'Inversión en IA por Región - {selected_year}',
            labels={'Investment': 'Inversión (miles de millones USD)', 'Entity': 'Región'},
            color='Investment',
            color_continuous_scale='Reds'
        )
        fig_bar.update_layout(
            xaxis_title="Región",
            yaxis_title="Inversión (miles de millones USD)",
            showlegend=False
        )
        st.plotly_chart(fig_bar, width='stretch')
    else:
        st.info(f"No hay datos de inversión por región para el año {selected_year} para mostrar en el gráfico de barras.")
    
    # Tabla con datos detallados
    st.subheader("Datos Detallados por Región")
    df_display = df_year.copy() # df_year might be empty
    if not df_display.empty:
        df_display['Investment'] = df_display['Investment'].round(1)
        st.dataframe(
            df_display,
            column_config={
                "Entity": "Región",
                "Year": "Año",
                "Investment": st.column_config.NumberColumn(
                    "Inversión (miles de millones USD)",
                    format="$%.1f"
                )
            },
            hide_index=True
        )
    else:
        st.info(f"No hay datos detallados por región para el año {selected_year} para mostrar en la tabla.")
    
    # Mostrar evolución temporal si hay múltiples años
    if len(years) > 1: # years comes from df_filtered, which is already checked for empty
        st.subheader("Evolución Temporal de la Inversión")
        fig_line = px.line(
            df_filtered, 
            x='Year', 
            y='Investment', 
            color='Entity',
            title='Evolución de la Inversión en IA por Región',
            labels={'Investment': 'Inversión (miles de millones USD)', 'Year': 'Año'}
        )
        fig_line.update_layout(
            xaxis_title="Año",
            yaxis_title="Inversión (miles de millones USD)"
        )
        st.plotly_chart(fig_line, width='stretch')