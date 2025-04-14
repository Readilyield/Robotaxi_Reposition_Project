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
YEARS = ['2021', '2022', '2023', '2024']
MONTHS = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
