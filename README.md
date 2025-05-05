## IEOR 290 - Robotaxi Reposition Project

The relocation route design for empty robotaxi is an important part to ensure an efficient and accessible robotaxi system in a complex network where rider demands varies across different regions.

This project focuses on the effect of fleet size and the tradeoff between maximizing service availability across all serviced regions and maximizing total revenue, which often priorities regions with high demand.<br>
This project applies the relocaiton matrix lookahead policy proposed by Braverman et al. (Empty-car Routing in Ridesharing Systems) and utilizes NYC taxi data to demonstrate the results.

### Data processing pipeline:
1. Download data and save in **nyc_trip/training data (or testing data)** directory <br>
   Download source [here](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) <br>
   (default `taxi_type` is `fhvhv`: for-hire vehicle high volume) <br>
   They should look like: `fhv_tripdata_2025-01.parquet`. <br>
#### In *nyc_trip/*
2. Open and run `Learning_Model_Params` to learn the parameters and compile data packages.
3. Open and run `Solve Relocation Matrix` to obtain fluid-based policy relocation matrix file.
4. Visualize your fluid-based policy relocation map in `Visualize_Relocation_Map`.
5. Now you should have all the processed data files you need and save them in *nyc_trip/*.

### Simulation pipeline:
tbd

#### Programming language and platform:
- Python
- JupyterNotebook
#### Required packages:
- main packages:
  | Numpy           | Matplotlib        | Networkx         | Scipy <br>

- other packages:
  | osmnx==2.0.1    | contextily==1.6.2 | gurobipy==12.0.1 | fastparquet==2024.11.0 <br>
  | pyarrow==19.0.1 | tqdm==4.67.1      | holidays==0.70   | seaborn==0.13.2 

#### File structure:
	|-> Notation  ::  Notations and formulations
	|- nyc_trip
	|- training data :: data parquets for training (to learn params and generate Q matrix)
    	|- testing data :: data parquets for testing
    	|-> Learning Model Params  ::  Learn parameter and constants in the fluid-based relocation policy
    	|-> Solve Relocation Matrix  ::  Solve the relocation matrix, you can change the formulation here
        |-> Visualize_Relocation_Map  ::  Visualize relocation policy over NYC regionalized map 
	|- simulate
    	|-> simulate  ::  Run simulation and evaluate the relocation matrix's effectiveness
    	|-> simulate.py  ::  Classes and objects for running the simulation
        |-> utils.py :: Helper functions to run multi-method simulation and display results 

