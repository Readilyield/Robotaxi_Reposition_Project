import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict

import holidays
from datetime import timedelta

from constants import (
    MAX_TAXI_ZONE_ID,
    location_ids,
    excluded_location_ids,
    location_id_to_index,
    num_locations,
    taxi_type,
    
    RIDER_ARRIVAL,
    RIDE_START,
    RIDER_LOST,
)
from relocation_policies import *
from simulate import TaxiSimulator

Delta = 20 # in minutes
T_max = int(24 * (60 // Delta))
YEARS = list(range(2019, 2025))

us_holidays = holidays.US(years=YEARS)

'''This function runs one simulation for each of:
   1. no relocation
   2. relocation with a pre-specified Q matrix
   3. shortest_wait policy 
   4. JLCR policy'''
def run_simulations_for_seed(
    seed: int,
    lambda_: np.ndarray,
    mu_: np.ndarray,
    P: np.ndarray,
    Q_base: np.ndarray,
    Q_path: str,
    arrival_events: list,
    T: int,
    R: int,
    N: int,
    max_time: float,
    eta: float = 0.5,
    output_dir: str = "sim_outputs"
):
    np.random.seed(seed)

    # sampling_modes = ["synthetic", "real"]
    sampling_modes =["real"]
    relocation_modes = {
        "no_reloc": {"policy": relocation_policy_blind_sampling, "Q": Q_base},
        "JLCR": {"policy": relocation_policy_jlcr_eta, "Q": Q_base},
        "shortest_wait": {"policy": relocation_policy_shortest_wait, "Q": Q_base},
        "fluidBased_policy": {"policy": relocation_policy_blind_sampling, "Q_path": Q_path},
    }

    for sampling in sampling_modes:
        use_real = sampling == "real"
        for reloc_key, config in relocation_modes.items():
            
            # Load Q matrix
            if "Q_path" in config:
                with np.load(config["Q_path"]) as data:
                    Q = data["Q"]
            else:
                Q = config["Q"]

            # Setup relocation policy
            policy = config["policy"]
            kwargs = {"eta": eta} if reloc_key == "JLCR" else {}

            # Init simulator
            sim = TaxiSimulator(
                T=T,
                R=R,
                N=N,
                lambda_=lambda_,
                mu_=mu_,
                P=P,
                Q=Q,
                relocation_policy=policy,
                relocation_kwargs=kwargs,
                use_real_demand=use_real,
                demand_events=arrival_events if use_real else None,
            )

            # Run sim
            sim.run(max_time=max_time)
            df_log = pd.DataFrame(sim.logger)

            # Save
            seed_dir = os.path.join(output_dir, str(seed))
            os.makedirs(seed_dir, exist_ok=True)
            fname = f"{sampling}Fix_demand__{reloc_key}.csv"
            df_log.to_csv(os.path.join(seed_dir, fname), index=False)
            
            print(f"[Seed {seed}] finished: {sampling} / {reloc_key}")


def prepare_arrival_events_from_real_data(df, num_days=3):
    """
    Given a pre-filtered NYC trip dataframe (weekdays, valid IDs, etc.),
    extract a simulation-ready list of rider arrival events across `num_days` 
    consecutive calendar days.

    Returns:
        events (List[Dict]): List of events with simulation time, 
        origin, destination, and trip_time (in hours).
    """
    unique_dates = df.pickup_datetime.dt.date.unique()
    working_days = [date for date in unique_dates if date.weekday() < 5 and date not in us_holidays]

    # Filter for weekdays that are NOT US holidays
    df = df[
        (df.pickup_datetime.dt.weekday < 5) &  # Monday to Friday
        (~df.pickup_datetime.dt.date.isin(us_holidays))  # Exclude US holidays
    ]

    # filter for valid locatino IDs
    df = df[df['PULocationID'].isin(location_ids) & df['DOLocationID'].isin(location_ids)]
    df['time_bin'] = (df['pickup_datetime'].dt.hour * (60 // Delta) + df['pickup_datetime'].dt.minute // Delta).astype(int)

    # Map IDs to array indices
    df['pu_idx'] = df['PULocationID'].map(location_id_to_index)
    df['do_idx'] = df['DOLocationID'].map(location_id_to_index)

    # Round pickup_datetime to dates only
    pickup_dates = df['pickup_datetime'].dt.date.to_numpy()

    # Find earliest set of consecutive calendar days
    unique_dates = np.unique(pickup_dates)
    for i in range(len(unique_dates) - num_days + 1):
        base = unique_dates[i]
        if all((base + timedelta(days=j)) in unique_dates for j in range(num_days)):
            selected_dates = {base + timedelta(days=j) for j in range(num_days)}
            break
    else:
        raise ValueError(f"No consecutive {num_days}-day window found.")

    # Filter df just once
    mask = np.isin(pickup_dates, list(selected_dates))
    df_sel = df.loc[mask].copy()

    # Sort once, for time order
    df_sel.sort_values('pickup_datetime', inplace=True)

    # Compute simulation time in hours relative to min time
    min_time = df_sel['pickup_datetime'].min()
    df_sel['t_sim'] = (df_sel['pickup_datetime'] - min_time).dt.total_seconds() / 3600.0

    # Convert trip_time to hours
    df_sel['trip_time_hr'] = df_sel['trip_time'] / 3600.0

    # Pack into list of event dicts
    return df_sel[['t_sim', 'pu_idx', 'do_idx', 'trip_time_hr']].to_dict('records')

def get_rider_arrival_timeseries(df_log, region_ids, bin_minutes=20):
    arrivals = df_log[df_log['event_type'] == RIDER_ARRIVAL].copy()
    arrivals['region'] = arrivals['data'].apply(lambda x: x['region'])
    arrivals = arrivals[arrivals['region'].isin(region_ids)]

    arrivals['time_bin'] = arrivals['datetime'].dt.floor(f'{bin_minutes}min')
    arrival_counts = arrivals.groupby('time_bin').size().reset_index(name='num_arrivals')

    return arrival_counts

def get_ridestarts_timeseries(df_log, region_ids, bin_minutes=20):
    ride_starts = df_log[df_log['event_type'] == RIDE_START].copy()
    ride_starts['region'] = ride_starts['data'].apply(lambda x: x['origin'])
    ride_starts = ride_starts[ride_starts['region'].isin(region_ids)]

    ride_starts['time_bin'] = ride_starts['datetime'].dt.floor(f'{bin_minutes}min')
    ride_starts = ride_starts.groupby('time_bin').size().reset_index(name='num_ridestarts')

    return ride_starts

def get_rider_lost_timeseries(df_log, region_ids, bin_minutes=20):
    lost_rides = df_log[df_log['event_type'] == RIDER_LOST].copy()
    lost_rides['region'] = lost_rides['data'].apply(lambda x: x['region'])
    lost_rides = lost_rides[lost_rides['region'].isin(region_ids)]

    lost_rides['time_bin'] = lost_rides['datetime'].dt.floor(f'{bin_minutes}min')
    lost_rides = lost_rides.groupby('time_bin').size().reset_index(name='num_lost_rides')

    return lost_rides

def fill_missing_time_bins(df_timeseries, start_time, end_time, bin_minutes=20, count_col='num_lost_rides'):
    full_time_index = pd.date_range(start=start_time, end=end_time, freq=f'{bin_minutes}min')
    full_df = pd.DataFrame({'time_bin': full_time_index})
    merged = full_df.merge(df_timeseries, on='time_bin', how='left')
    merged[count_col] = merged[count_col].fillna(0).astype(int)
    return merged

def plot_arrival_versus_ridestarts(df_log, region_id, bin_minutes):
    arrival_ts = get_rider_arrival_timeseries(df_log, region_id=region_id, bin_minutes=bin_minutes)
    ridestarts_ts = get_ridestarts_timeseries(df_log, region_id=region_id, bin_minutes=bin_minutes)
    lostrides_ts = get_rider_lost_timeseries(df_log, region_id=region_id, bin_minutes=bin_minutes)


    start_time = arrival_ts['time_bin'].min() + timedelta(days=2) # ignore first two days for system to reach stationary distribution
    end_time = arrival_ts['time_bin'].max()

    arrival_ts = fill_missing_time_bins(
        arrival_ts,
        start_time=start_time,
        end_time=end_time,
        bin_minutes=bin_minutes,
        count_col='num_arrivals'
    )

    ridestarts_ts = fill_missing_time_bins(
        ridestarts_ts,
        start_time=start_time,
        end_time=end_time,
        bin_minutes=bin_minutes,
        count_col='num_ridestarts'
    )
    
    lostrides_ts = fill_missing_time_bins(
        lostrides_ts,
        start_time=start_time,
        end_time=end_time,
        bin_minutes=bin_minutes,
        count_col='num_lost_rides'
    )
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(arrival_ts['time_bin'], arrival_ts['num_arrivals'], label='Rider Arrivals', color='blue')
    plt.plot(ridestarts_ts['time_bin'], ridestarts_ts['num_ridestarts'], label='Ride Starts', color='orange')
    plt.plot(lostrides_ts['time_bin'], lostrides_ts['num_lost_rides'], label='Riders Lost', color='purple')
    plt.title('Rider Arrivals/Starts/Lost Rides Over Time')
    plt.xlabel('Time')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

######################
###################
'''PLot simulation'''
##################
#####################

def plot_simulation_grid_for_strategies(
    demand_mode: str,
    strategies: list,
    region_ids: list,
    bin_minutes: int = 20,
    log_dir: str = "sim_outputs",
    start_time: str = '2025-01-02'
):
    """
    Plot rider arrival, ride start, and lost rider time series (binned)
    for each relocation strategy under a specific demand mode.
    
    Args:
        demand_mode (str): 'real' or 'synthetic'
        strategies (list): list of strategy names used in file naming
        region_id (int): region ID to analyze
        bin_minutes (int): time bin size in minutes
        log_dir (str): directory where logs are saved
    """

    # Internal helper
    def load_and_prepare_log(filepath):
        df = pd.read_csv(filepath, converters={'data': eval})
        df.dropna(how='any')

        df['datetime'] = pd.to_timedelta(df['time'], unit='h') + pd.Timestamp(start_time)

        return df

    fig, axes = plt.subplots(2, 2, figsize=(20, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    sns.set(style="whitegrid")

    for i, strategy in enumerate(strategies):
        fname = f"{demand_mode}_demand__{strategy}.csv"
        path = os.path.join(log_dir, fname)

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        df_log = load_and_prepare_log(path)

        arrival_ts = get_rider_arrival_timeseries(df_log, region_ids=region_ids, bin_minutes=bin_minutes)
        ridestarts_ts = get_ridestarts_timeseries(df_log, region_ids=region_ids, bin_minutes=bin_minutes)
        lostrides_ts = get_rider_lost_timeseries(df_log, region_ids=region_ids, bin_minutes=bin_minutes)

        if arrival_ts.empty:
            print(f"No arrival data for strategy {strategy}.")
            continue

        # start_time = arrival_ts['time_bin'].min() + timedelta(days=2)
        start_time = arrival_ts['time_bin'].min()
        end_time = arrival_ts['time_bin'].max()

        arrival_ts = fill_missing_time_bins(arrival_ts, start_time, end_time, bin_minutes, count_col='num_arrivals')
        ridestarts_ts = fill_missing_time_bins(ridestarts_ts, start_time, end_time, bin_minutes, count_col='num_ridestarts')
        lostrides_ts = fill_missing_time_bins(lostrides_ts, start_time, end_time, bin_minutes, count_col='num_lost_rides')

        ax = axes[i]
        ax.plot(arrival_ts['time_bin'], arrival_ts['num_arrivals'], label='Arrivals', color='blue')
        ax.plot(ridestarts_ts['time_bin'], ridestarts_ts['num_ridestarts'], label='Ride Starts', color='orange')
        ax.plot(lostrides_ts['time_bin'], lostrides_ts['num_lost_rides'], label='Lost Riders', color='purple')
        ax.set_title(strategy.replace("_", " ").upper())
        ax.tick_params(axis='x', rotation=45)
        if i % 2 == 0:
            ax.set_ylabel("Count")
        if i >= 4:
            ax.set_xlabel("Time")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='best', ncol=3)
    if len(region_ids) == 1:
        fig.suptitle(
            f"Rider Arrivals, Ride Starts, and Lost Riders\nRegion {region_ids[0]} — {demand_mode.capitalize()} Demand",
            fontsize=16
        )
    else:
        fig.suptitle(
            f"Rider Arrivals, Ride Starts, and Lost Riders\n {len(region_ids)} Regions — {demand_mode.capitalize()} Demand",
            fontsize=16
        )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def compute_system_metrics(df_log: pd.DataFrame, T: int, R: int, N: int) -> dict:
    """
    Computes 8 key system-wide metrics from a simulation log.

    Parameters:
        df_log (pd.DataFrame): simulation log dataframe
        T (int): number of time blocks (e.g., 72)
        R (int): number of regions
        N (int): number of vehicles

    Returns:
        dict[str, float]: dictionary of scalar metric values
    """

    # --- 1. Basic counts ---
    arrivals = df_log[df_log['event_type'] == 'rider_arrival'].copy()
    ride_starts = df_log[df_log['event_type'] == 'ride_start'].copy()
    rider_lost = df_log[df_log['event_type'] == 'rider_lost'].copy()
    reloc_starts = df_log[df_log['event_type'] == 'relocation_start'].copy()

    # All durations in hours
    ride_completions = df_log[df_log['event_type'] == 'ride_completion'].copy()
    relocation_completions = df_log[df_log['event_type'] == 'relocation_completion'].copy()

    total_sim_time = df_log['time'].max()  # hours
    total_vehicle_time = N * total_sim_time

    # --- 2. Abandonment + Fulfillment ---
    num_arrivals = len(arrivals)
    num_ride_starts = len(ride_starts)
    num_lost = len(rider_lost)

    abandonment_rate = num_lost / num_arrivals if num_arrivals > 0 else 0.0
    fulfillment_rate = num_ride_starts / num_arrivals if num_arrivals > 0 else 0.0

    # --- 3. Vehicle time usage (utilization, idle, relocation) ---
    ride_durations = ride_completions['data'].apply(lambda x: x.get('travel_time', 0))
    relocation_durations = relocation_completions['data'].apply(lambda x: x.get('travel_time', 0))

    mean_ride_time = ride_durations.mean()
    mean_relocation_time = relocation_durations.mean()

    utilization_rate = mean_ride_time / total_sim_time
    idle_vehicle_fraction = 1 - (mean_ride_time + mean_relocation_time) / total_sim_time
    empty_relocation_ratio = mean_relocation_time /total_sim_time

    # --- 4. Peak abandonment over time bins ---
    arrivals['region'] = arrivals['data'].apply(lambda x: x['region'])
    arrivals['time_bin'] = arrivals['datetime'].dt.floor('20min')
    lost = rider_lost.copy()
    lost['region'] = lost['data'].apply(lambda x: x['region'])
    lost['time_bin'] = lost['datetime'].dt.floor('20min')

    arrival_counts = arrivals.groupby('time_bin').size()
    lost_counts = lost.groupby('time_bin').size()

    timebin_abandonment = (lost_counts / arrival_counts).fillna(0)
    peak_abandonment_rate = timebin_abandonment.max()

    # --- 5. Relocations per vehicle per hour ---
    relocations_per_vehicle_per_hour = len(reloc_starts) / total_vehicle_time

    # --- 6. Regional fairness metrics ---
    arrivals_by_region = arrivals.groupby('region').size()
    ride_starts['region'] = ride_starts['data'].apply(lambda x: x['origin'])
    starts_by_region = ride_starts.groupby('region').size()
    fulfillment_by_region = (starts_by_region / arrivals_by_region).fillna(0)

    max_abandonment_region = 1 - fulfillment_by_region.min()
    std_fulfillment_across_regions = fulfillment_by_region.std()

    return {
        'abandonment_rate': round(abandonment_rate, 4),
        'fulfillment_rate': round(fulfillment_rate, 4),
        'mean_utilization_rate': round(utilization_rate, 4),
        'mean_idle_vehicle_fraction': round(idle_vehicle_fraction, 4),
        'mean_empty_relocation_ratio': round(empty_relocation_ratio, 4),
        'peak_abandonment_rate': round(peak_abandonment_rate, 4),
        'relocations_per_vehicle_per_hour': round(relocations_per_vehicle_per_hour, 4),
        'max_regional_abandonment_rate': round(max_abandonment_region, 4),
        'std_fulfillment_across_regions': round(std_fulfillment_across_regions, 4),
    }

def summarize_multiple_runs(setting_name: str, 
                            T: int, R: int, N: int, 
                            base_dir: str = "sim_outputs") -> pd.DataFrame:
    """
    Summarize multiple runs of the same simulation setting.

    Args:
        setting_name (str): e.g., 'synthetic_demand__JLCR.csv'
        T (int): number of time blocks
        R (int): number of regions
        N (int): number of vehicles
        base_dir (str): parent directory holding subfolders 0/, 1/, ..., each with logs

    Returns:
        pd.DataFrame: with metrics as rows, and columns ['mean', 'std']
    """
    metrics_list = []
    run_dirs = sorted([d for d in os.listdir(base_dir) 
                       if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()])

    for run_id in run_dirs:
        filepath = os.path.join(base_dir, run_id, setting_name)
        if not os.path.exists(filepath):
            print(f"Skipping missing file: {filepath}")
            continue

        df_log = pd.read_csv(filepath, converters={"data": eval})
        df_log['datetime'] = pd.to_timedelta(df_log['time'], unit='h') + pd.Timestamp("2025-01-02")

        metrics = compute_system_metrics(df_log, T=T, R=R, N=N)
        metrics_list.append(metrics)

    if not metrics_list:
        raise ValueError("No valid simulation runs found.")

    df_metrics = pd.DataFrame(metrics_list)
    df_summary = df_metrics.agg(['mean', 'std']).T
    df_summary.columns = ['mean', 'std']
    return df_summary.round(4)


def run_all_simulations_for_seed(
    seed: int,
    lambda_: np.ndarray,
    mu_: np.ndarray,
    P: np.ndarray,
    Q_base: np.ndarray,
    arrival_events: list,
    T: int,
    R: int,
    N: int,
    max_time: float,
    eta: float = 0.5,
    output_dir: str = "sim_outputs"
):
    np.random.seed(seed)

    sampling_modes = ["synthetic", "real"]
    relocation_modes = {
        "no_reloc": {"policy": relocation_policy_blind_sampling, "Q": Q_base},
        "JLCR": {"policy": relocation_policy_jlcr_eta, "Q": Q_base},
        "shortest_wait": {"policy": relocation_policy_shortest_wait, "Q": Q_base},
        "Q_2": {"policy": relocation_policy_blind_sampling, "Q_path": "../nyc_trip/Qs_2.npz"},
        "Q_4": {"policy": relocation_policy_blind_sampling, "Q_path": "../nyc_trip/Qs_4.npz"},
        "Q_8": {"policy": relocation_policy_blind_sampling, "Q_path": "../nyc_trip/Qs_8.npz"},
    }

    for sampling in sampling_modes:
        use_real = sampling == "real"
        for reloc_key, config in relocation_modes.items():
            
            
            # Load Q matrix
            if "Q_path" in config:
                with np.load(config["Q_path"]) as data:
                    Q = data["Qs"]
            else:
                Q = config["Q"]

            # Setup relocation policy
            policy = config["policy"]
            kwargs = {"eta": eta} if reloc_key == "JLCR" else {}

            # Init simulator
            sim = TaxiSimulator(
                T=T,
                R=R,
                N=N,
                lambda_=lambda_,
                mu_=mu_,
                P=P,
                Q=Q,
                relocation_policy=policy,
                relocation_kwargs=kwargs,
                use_real_demand=use_real,
                demand_events=arrival_events if use_real else None,
            )

            # Run sim
            sim.run(max_time=max_time)
            df_log = pd.DataFrame(sim.logger)

            # Save
            seed_dir = os.path.join(output_dir, str(seed))
            os.makedirs(seed_dir, exist_ok=True)
            fname = f"{sampling}_demand__{reloc_key}.csv"
            df_log.to_csv(os.path.join(seed_dir, fname), index=False)
            
            print(f"[Seed {seed}] finished: {sampling} / {reloc_key}")