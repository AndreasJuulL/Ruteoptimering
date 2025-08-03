import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import googlemaps

# === Tidsfunktioner ===
def time_to_minutes(t):
    if pd.isna(t):
        return 0
    if isinstance(t, str):
        dt = datetime.strptime(t, "%H:%M")
        return dt.hour * 60 + dt.minute
    elif isinstance(t, (int, float)):
        return int(t * 24 * 60)
    else:
        raise ValueError(f"Uventet tidsformat: {t}")

def minutes_to_time_str(minutes):
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02}:{mins:02}"

# === Google Maps Matrix ===
def get_batched_time_matrix(addresses, gmaps, max_elements=100):
    n = len(addresses)
    time_matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        max_destinations = max_elements // 25
        for j_start in range(0, n, max_destinations):
            j_end = min(j_start + max_destinations, n)
            destinations = addresses[j_start:j_end]

            try:
                result = gmaps.distance_matrix(
                    origins=[addresses[i]],
                    destinations=destinations,
                    mode="driving",
                    departure_time="now",
                    traffic_model="best_guess"
                )
                for k, element in enumerate(result["rows"][0]["elements"]):
                    duration = element.get("duration")
                    seconds = duration["value"] if duration else 999999
                    time_matrix[i][j_start + k] = seconds // 60
            except Exception as e:
                st.error(f"❌ Fejl under kald {i} → {j_start}-{j_end}: {e}")
                for k in range(j_start, j_end):
                    time_matrix[i][k] = 999999

            time.sleep(1)

    return time_matrix.tolist()

# === Datamodel ===
def create_data_model(df, indstillinger_df, gmaps):
    if "Aktiv" in df.columns:
        df = df[df["Aktiv"] == True].copy()
        if df.empty:
            st.error("Ingen aktive stop fundet i regnearket.")
            return None

    df["RuteNr"] = df.get("RuteNr", pd.Series([None]*len(df)))

    try:
        num_vehicles = int(Indstillinger_df.loc["AntalKøretøjer", "Værdi"])
    except Exception:
        st.warning("⚠️ 'AntalKøretøjer' ikke fundet – bruger 1 som standard.")
        num_vehicles = 10

    stop_names = df["Navn"].tolist()
    addresses = df["Adresse"].tolist()
    stop_times = df["StopTid"].fillna(0).astype(int).tolist()

    start_tider = df.get("StartTid", pd.Series([None]*len(df)))
    slut_tider = df.get("SlutTid", pd.Series([None]*len(df)))

    time_windows = []
    for start, end in zip(start_tider, slut_tider):
        start_min = time_to_minutes(start) if pd.notna(start) else 0
        end_min = time_to_minutes(end) if pd.notna(end) else 1440
        if start_min > end_min:
            end_min = start_min
        time_windows.append((start_min, end_min))

    try:
        start_index = stop_names.index("Køkken")
    except ValueError:
        st.error("⚠️ 'Køkken' ikke fundet blandt aktive stop. Tilføj den og marker som aktiv.")
        return None

    time_matrix = get_batched_time_matrix(addresses, gmaps)
    time_windows[start_index] = (0, 1440)

    return {
        "df": df,
        "stop_names": stop_names,
        "addresses": addresses,
        "stop_times": stop_times,
        "time_matrix": time_matrix,
        "time_windows": time_windows,
        "num_vehicles": num_vehicles,
        "starts": [start_index] * num_vehicles,
        "ends": [start_index] * num_vehicles,
        "start_index": start_index
    }

# === Ruteoptimering ===
def run_routing(data):
    manager = pywrapcp.RoutingIndexManager(
        len(data['time_matrix']),
        data['num_vehicles'],
        data['starts'],
        data['ends']
    )
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        travel_time = data['time_matrix'][from_node][to_node]
        service_time = data['stop_times'][from_node]
        return travel_time + service_time

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.AddDimension(
        transit_callback_index, 30, 1440, False, "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    for i, (start, end) in enumerate(data['time_windows']):
        index = manager.NodeToIndex(i)
        time_dimension.CumulVar(index).SetRange(start, end)

    for vehicle_id in range(data['num_vehicles']):
        idx = routing.Start(vehicle_id)
        time_dimension.CumulVar(idx).SetRange(560, 570)

    df = data["df"]
    for idx, row in df.iterrows():
        if row["Navn"] == "Køkken":
            continue
        node_index = manager.NodeToIndex(idx)
        rute_nr = row["RuteNr"]
        if pd.notna(rute_nr):
            try:
                vehicle = int(rute_nr) - 1
                if vehicle < data["num_vehicles"]:
                    routing.SetAllowedVehiclesForIndex([vehicle], node_index)
            except:
                pass

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.time_limit.seconds = 60

    solution = routing.SolveWithParameters(search_parameters)
    return solution, manager, routing, time_dimension

# === Formatering ===
def format_solution(data, solution, manager, routing, time_dimension):
    output = ""
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            arrival = solution.Min(time_dimension.CumulVar(index))
            route.append((node, arrival))
            index = solution.Value(routing.NextVar(index))
        node = manager.IndexToNode(index)
        arrival = solution.Min(time_dimension.CumulVar(index))
        route.append((node, arrival))

        output += f"### Rute for køretøj {vehicle_id + 1}:\n\n"
        maps_link = "https://www.google.com/maps/dir/" + "/".join(
            data['addresses'][node].replace(" ", "+") for node, _ in route
        ) + "/\n"

        for node, arrival in route:
            navn = data["stop_names"][node]
            stoptid = data["stop_times"][node]
            departure = arrival + stoptid
            output += f"- {navn} ({minutes_to_time_str(arrival)} - {minutes_to_time_str(departure)})\n"
        output += f"\n[Google Maps rute-link]({maps_link})\n\n"

    return output

# === Streamlit UI ===
st.title("📍 Ruteoptimering med Google Maps")

uploaded_file = st.file_uploader("Upload Excel-arket (.xlsx)", type=["xlsx"])

if uploaded_file:
    if "GOOGLE_MAPS_API_KEY" not in st.secrets:
        st.error("API-nøglen mangler i .streamlit/secrets.toml.")
    else:
        gmaps = googlemaps.Client(key=st.secrets["GOOGLE_MAPS_API_KEY"])

        try:
            df = pd.read_excel(uploaded_file, sheet_name=None)
            if "Indstillinger" not in df:
                st.error("Ark 'Indstillinger' mangler i regnearket.")
            else:
                stops_df = df[next(iter(df))]  # Første ark antages som stops
                indstillinger_df = df["Indstillinger"]

                with st.spinner("🔄 Henter rejsetider..."):
                    data = create_data_model(stops_df, indstillinger_df, gmaps)

                if data:
                    with st.spinner("🚚 Optimerer ruter..."):
                        solution, manager, routing, time_dimension = run_routing(data)

                    if solution:
                        st.success("✅ Løsning fundet!")
                        st.markdown(format_solution(data, solution, manager, routing, time_dimension))
                    else:
                        st.error("❌ Ingen løsning fundet.")
        except Exception as e:
            st.error(f"Fejl ved læsning af regnearket: {e}")
else:
    st.info("Upload venligst en Excel-fil for at starte.")
