## IEOR 290 - Robotaxi Reposition Project

The relocation route design for empty robotaxi is an important part to ensure an efficient and accessible robotaxi system in a complex network where rider demands varies across different regions.

This project focuses on the effect of fleet size and the tradeoff between maximizing service availability across all serviced regions and maximizing total revenue, which often priorities regions with high demand.<br>
This project applies the relocaiton matrix lookahead policy proposed by Braverman et al. (Empty-car Routing in Ridesharing Systems) and utilizes NYC taxi data to demonstrate the results.


#### Programming language and platform:
- Python
- JupyterNotebook
#### Required packages:
- Numpy
- Matplotlib
- Networkx
- optional:<br>
  | osmnx==2.0.1    | contextily==1.6.2 | gurobipy==12.0.1 | fastparquet==2024.11.0 <br>
  | pyarrow==19.0.1 | tqdm==4.67.1      | holidays==0.70

#### File structure:
	|-> Notation  ::  Notations and formulations
	|- nyc_trip
    	|-> Constructing NYC Taxi Grid  ::  Visualize relocation in the NYC map 
    	|-> Learning Model Params  ::  Learn parameters of the relocation algo from NYC taxi data
    	|-> Solve Relocation Matrix  ::  Solve relocation matrix, we can mod constraints here
    	|-> Qs.npz  ::  solved relocation matrix
	|- simulate
    	|-> simulate  ::  Run simulation and evaluate the relocation matrix's effectiveness
    	|-> simulate.py  ::  Classes and objects for running the simulation 

