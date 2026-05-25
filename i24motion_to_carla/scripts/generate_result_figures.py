import i24_motion_data
import pandas
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plotTimeDiagramFromI24Data(csv_path, start_timestamp, road, selected_lane, road_name, figure_path):
    mpl.rcParams['font.size'] = 24
    sim_data = pandas.read_csv(csv_path)
    total_timestep = float(sim_data["simulation_time"].max())
    min_s = float(sim_data[sim_data["hero_status"] == 1]["s"].min())
    max_s = float(sim_data[sim_data["hero_status"] == 1]["s"].max())
    source_data = i24_motion_data.I24MotionData(road, 1669812000.0 , 1669812600, -1600, 1600)
    print(start_timestamp, start_timestamp+total_timestep, min_s, max_s)
    edie_box_sample = source_data.queryEdieBoxSubset(start_timestamp, start_timestamp+total_timestep, min_s, max_s)
    sim_data_filtered = sim_data[(sim_data["road_id"] == road)]
    if selected_lane is not None:
        sim_data_filtered = sim_data[(sim_data["lane_id"] == selected_lane)]
    unique_ids = list(sim_data_filtered["vehicle_id"].unique())
    fig, ax = plt.subplots(figsize=(16, 12))
    for unique in unique_ids:
        data_unique = sim_data_filtered[sim_data_filtered["vehicle_id"] == unique]
        if data_unique["hero_status"].max() >= 0.5:
            ax.plot(data_unique["simulation_time"], data_unique["s"], linewidth=4.0, color='red', label="Hero Vehicle")
    print(edie_box_sample)
    for lane in edie_box_sample:
        if (selected_lane is not None) and (lane != selected_lane):
            continue
        edie_box_sample_lane = edie_box_sample[lane]
        unique_values = edie_box_sample_lane["id"].unique()
        for entry_id in unique_values:
            entry = edie_box_sample_lane[edie_box_sample_lane["id"] == entry_id]
            ax.plot(entry["time"] - start_timestamp, entry["s"], linewidth=1.0, alpha=0.5, color='black', label="I24 Real Vehicle")
    ax.set_title(f"{road_name} Road", fontsize=32)
    ax.legend(handles=[
        mpatches.Patch(color='red', label="Hero Vehicle"),
        mpatches.Patch(color='black', label="I24 Real Vehicle")
    ], fontsize=32, loc="upper left")
    ax.set_xlabel("Time (s)", fontsize=32)
    ax.set_ylabel("Longitudinal Road Position (m)", fontsize=32)
    fig.savefig(figure_path)

plotTimeDiagramFromI24Data("result_new_road_1_lane_2.csv", 1669812350, 1, None, "Eastbound", "eastbound_timespace.png")
plotTimeDiagramFromI24Data("result_new_road_2_lane_1.csv", 1669812350, 2, None, "Westbound", "westbound_timespace.png")