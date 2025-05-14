MAX_TAXI_ZONE_ID = 265
location_ids = range(1, MAX_TAXI_ZONE_ID+1)
excluded_location_ids = [
    # We exclude the following locations:
    # 1. Middle of nowheres
    # 2. EWR Airport
    # 3. Islands (except Roosevelt Island)
    
    # Staten Island
    5,
    6,
    23,
    44,
    84,
    99, 
    109,
    110,
    115,
    118,
    156,
    172,
    176,
    187,
    204,
    206,
    214,
    221,
    245,
    251,
    
    # Ellis Island
    103,
    104,
    105,
    46, # Bronx City Island
    1, # EWR Airport (Ridesharing app pickups and dropoffs at EWR are banned.)
    2, # Jamaica Bay
    194, # Randalls Island
    264, # Unknown
    265, # Outside NYC
    179, # Rikers Island
    199, # Rikers Island
    ]

location_ids = [id_ for id_ in location_ids if id_ not in excluded_location_ids]
# Create a mapping from location IDs to indices
location_id_to_index = {id_: i for i, id_ in enumerate(location_ids)}
num_locations = len(location_ids)

taxi_type = 'fhvhv' # 'green', 'fhv', 'fhvhv'


import numpy as np 
def get_region_trip_distribution(region_id, trip_counts, location_id_to_index):
    """
    Calculates the distribution of trips going into and out of a specific region_id,
    and identifies the top 10 regions with the highest total interaction.

    Args:
        region_id (int): The original location ID (PULocationID/DOLocationID) to analyze.
        trip_counts (np.ndarray): The 3D numpy array of trip counts
                                  (time_bin, pu_idx, do_idx).
        location_id_to_index (dict): Mapping from original location ID to index.

    Returns:
        tuple: A tuple containing:
            - outgoing_distribution (dict): Distribution of trips from region_id
                                            to other region_ids {target_id: count}.
            - incoming_distribution (dict): Distribution of trips from other
                                            region_ids to region_id {source_id: count}.
            - top_10_interacting_regions (list): List of tuples [(region_id, total_interaction_count)]
                                                  for the top 10 other regions by total trips
                                                  (incoming + outgoing).
            - None: If the region_id is not found in the mapping.
    """
    if region_id not in location_id_to_index:
        print(f"Error: Region ID {region_id} not found in location ID mapping.")
        return None, None, None

    region_idx = location_id_to_index[region_id] #mapped location id, may not be the same as on-map region id
    
    num_locations = trip_counts.shape[1] # Or trip_counts.shape[2]

    # --- Calculate Outgoing Distribution ---
    outgoing_counts = np.sum(trip_counts[:, region_idx, :], axis=0)

    # Convert index-based counts to a dictionary using original region IDs
    outgoing_distribution = {}
    for do_idx in range(num_locations):
        do_idx
        if outgoing_counts[do_idx] > 0: # Only include regions with actual trips
             outgoing_distribution[do_idx] = int(outgoing_counts[do_idx]) # Convert to int

    # --- Calculate Incoming Distribution ---
    incoming_counts = np.sum(trip_counts[:, :, region_idx], axis=0)

    # Convert index-based counts to a dictionary using original region IDs
    incoming_distribution = {}
    for pu_idx in range(num_locations):
        pu_idx
        if incoming_counts[pu_idx] > 0: # Only include regions with actual trips
             incoming_distribution[pu_idx] = int(incoming_counts[pu_idx]) # Convert to int


    # --- Identify Top 10 Interacting Regions ---
    trip_counts_total = np.sum(trip_counts, axis=0) # Shape (num_locations, num_locations)

    interaction_counts = []
    for other_idx in range(num_locations):
        if other_idx == region_idx:
            continue # Skip interaction with itself for "other regions"

        # Total trips from region_idx to other_idx PLUS from other_idx to region_idx
        total_interaction = trip_counts_total[region_idx, other_idx] + trip_counts_total[other_idx, region_idx]

        if total_interaction > 0: # Only consider regions with interaction
             interaction_counts.append((total_interaction, other_idx))

    # Sort interactions by count in descending order
    interaction_counts.sort(key=lambda item: item[0], reverse=True)

    # Take the top 10
    top_10_interacting_regions = [(region_id, count) for count, region_id in interaction_counts[:10]]

    return outgoing_distribution, incoming_distribution, top_10_interacting_regions

