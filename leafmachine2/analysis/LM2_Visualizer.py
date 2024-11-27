import streamlit as st
import sqlite3, random
import polars as pl
import plotly.express as px
from streamlit_plotly_events import plotly_events

# Path to SQLite database (static)
DB_PATH = '/media/nas/GBIF_Downloads/Combined_LM2_Data/lm2_vault.db'

class LM2DataVisualizer:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_data_from_db():
        """
        Query the SQLite database for data and return a Polars DataFrame.
        Caching this to avoid querying the database repeatedly.
        """
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM vault"
        df = pl.read_database(query, conn)
        conn.close()
        return df

    @staticmethod
    @st.cache_data(ttl=600)
    def get_unique_values_from_db():
        """
        Query the SQLite database for unique values from the unique_values table.
        Caching this to avoid recalculating the unique values repeatedly.
        """
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM unique_values"
        df = pl.read_database(query, conn)
        conn.close()

        # Get unique values for each column
        unique_values = {}
        org_columns = df.select("org_column").unique().to_series().to_list()

        for col in org_columns:
            unique_values[col] = df.filter(pl.col("org_column") == col).select("unique_value").to_series().to_list()

        return unique_values

    @staticmethod
    def filter_data_by_taxonomy(df, filters):
        """
        Filter the Polars DataFrame based on taxonomy hierarchy.
        Start by applying org_family -> org_genus -> org_fullname filter cascade.
        """
        # Apply org_family filter first
        if filters['org_family']:
            df = df.filter(pl.col('org_family').is_in(filters['org_family']))

        # Apply org_genus filter if a family is already selected
        if filters['org_genus']:
            df = df.filter(pl.col('org_genus').is_in(filters['org_genus']))

        # Apply org_fullname filter as the main class filter
        if filters['org_fullname']:
            df = df.filter(pl.col('org_fullname').is_in(filters['org_fullname']))

        # Apply org_herbcode filter independently
        if filters['org_herbcode']:
            df = df.filter(pl.col('org_herbcode').is_in(filters['org_herbcode']))

        return df

    @staticmethod
    def apply_sample(df, sample_percentage):
        """
        Randomly sample the filtered Polars DataFrame based on the specified percentage.
        """
        if 0 < sample_percentage < 1:
            return df.sample(fraction=sample_percentage, seed=random.randint(1, 1000))
        return df

    @staticmethod
    @st.cache_data(ttl=600)
    def count_unique_entries():
        """
        Query the SQLite database to count the number of unique entries
        for 'filename', 'component_name', and 'org_herbcode' columns.
        Returns a dictionary with counts for each.
        """
        conn = sqlite3.connect(DB_PATH)

        # Load the table into a Polars DataFrame
        query = "SELECT filename, component_name, org_herbcode FROM vault"
        df = pl.read_database(query, conn)
        conn.close()

        # Compute unique counts
        counts = {
            'unique_filenames': df.select(pl.col("filename")).unique().height,
            'unique_component_names': df.select(pl.col("component_name")).unique().height,
            'unique_herbarium_codes': df.select(pl.col("org_herbcode")).unique().height
        }

        return counts

def apply_filters():
    """Apply the filters based on user selection."""
    filters = {}
    for col in org_columns:
        filters[col] = st.session_state[f'{col}_filter']
    return LM2DataVisualizer.filter_data_by_taxonomy(data, filters)



##############################################################
# Initialize the app
st.title("LM2 Data Vault Visualizer")

# Load data from the database
data = LM2DataVisualizer.get_data_from_db()

# Define the columns we want to use for filtering (based on `org_` columns)
org_columns = ['org_herbcode', 'org_family', 'org_genus', 'org_fullname']

# Get unique values from the database's unique_values table
unique_values = LM2DataVisualizer.get_unique_values_from_db()

# Initialize session state for filters and ensure it's initialized correctly
if 'filters' not in st.session_state:
    st.session_state.filters = {col: [] for col in org_columns}


# X and Y axis variable options
plot_variables = [
    'image_height', 'image_width', 'n_pts_in_polygon', 'conversion_mean', 
    'predicted_conversion_factor_cm', 'area', 'perimeter', 'convex_hull', 
    'rotate_angle', 'bbox_min_long_side', 'bbox_min_short_side', 'convexity', 
    'concavity', 'circularity', 'aspect_ratio', 'angle', 'distance_lamina', 
    'distance_width', 'distance_petiole', 'distance_midvein_span', 'distance_petiole_span', 
    'trace_midvein_distance', 'trace_petiole_distance', 'apex_angle', 'base_angle'
]


# Button to fetch and display unique counts
if st.button("Show Unique Counts"):
    unique_counts = LM2DataVisualizer.count_unique_entries()
    
    st.write(f"Unique Filename Entries: {unique_counts['unique_filenames']}")
    st.write(f"Unique Component Names: {unique_counts['unique_component_names']}")
    st.write(f"Unique Herbarium Codes: {unique_counts['unique_herbarium_codes']}")

    
# Primary class selection
primary_class = st.selectbox("Select Primary Class", org_columns)

# Sample percentage
sample_percentage = st.slider("Sample percentage", 0.0, 1.0, 1.0, step=0.05)


# Filter selection: each filter is in its own expander
with st.expander(f"Filter by org_herbcode"):
    st.session_state['org_herbcode_filter'] = st.multiselect(
        "Select org_herbcode", unique_values['org_herbcode'], default=unique_values['org_herbcode'])

with st.expander(f"Filter by org_family"):
    st.session_state['org_family_filter'] = st.multiselect(
        "Select org_family", unique_values['org_family'], default=unique_values['org_family'])

if st.session_state['org_family_filter']:
    filtered_genus_values = data.filter(pl.col('org_family').is_in(st.session_state['org_family_filter'])).select("org_genus").unique().to_series().to_list()
else:
    filtered_genus_values = unique_values['org_genus']

with st.expander(f"Filter by org_genus"):
    st.session_state['org_genus_filter'] = st.multiselect(
        "Select org_genus", filtered_genus_values, default=filtered_genus_values)

if st.session_state['org_genus_filter']:
    filtered_fullname_values = data.filter(pl.col('org_genus').is_in(st.session_state['org_genus_filter'])).select("org_fullname").unique().to_series().to_list()
else:
    filtered_fullname_values = unique_values['org_fullname']

with st.expander(f"Filter by org_fullname"):
    st.session_state['org_fullname_filter'] = st.multiselect(
        "Select org_fullname", filtered_fullname_values, default=filtered_fullname_values)

# Button to apply filters
if st.button("Apply Filters"):
    filtered_data = apply_filters()
    filtered_data = LM2DataVisualizer.apply_sample(filtered_data, sample_percentage)
    st.session_state['filtered_data'] = filtered_data

# Plot type selection
plot_type = st.selectbox("Select Plot Type", ["Violin Plot", "Scatter Plot"])

# Plot selection (X and Y axis variables)
x_var = st.selectbox("Select X-axis", plot_variables)

# Show Y-axis selectbox only if Scatter Plot is selected
if plot_type == "Scatter Plot":
    y_var = st.selectbox("Select Y-axis", plot_variables)

# Button to create the plot
if st.button("Create Plot"):
    if 'filtered_data' not in st.session_state or st.session_state['filtered_data'].empty:
        st.warning("No data available to plot. Apply filters first!")
    else:
        filtered_data = st.session_state['filtered_data']

        # Dynamically adjust plot height based on the number of unique classes
        num_classes = filtered_data[primary_class].nunique()
        fig_height = max(400, num_classes * 50)  # Set a minimum height of 400px

        # Show the plots based on selection
        if plot_type == "Violin Plot":
            fig = px.violin(filtered_data, x=x_var, color=primary_class, box=True, points="all", range_y=[0, None])
            fig.update_layout(height=fig_height)  # Dynamically adjust the height
            selected_points = plotly_events(fig)
            st.plotly_chart(fig)

        elif plot_type == "Scatter Plot":
            fig = px.scatter(filtered_data, x=x_var, y=y_var, color=primary_class, 
                             hover_data=['org_genus', 'org_species'])
            fig.update_layout(height=fig_height)  # Dynamically adjust the height
            selected_points = plotly_events(fig, click_event=True, hover_event=False)
            st.plotly_chart(fig)

        # Optional: Show details of selected points from Plotly events
        if selected_points:
            st.write("Selected Points Data:")
            st.write(selected_points)