## IEOR 290 - Robotaxi Reposition Project

The relocation route design for empty robotaxi is an important part to ensure an efficient and accessible robotaxi system in a complex network where rider demands varies across different regions.
This project focuses on the effect of fleet size and the tradeoff between maximizing service availability across all serviced regions and maximizing total revenue, which often priorities regions with high demand.
This project applies the relocaiton matrix lookahead policy proposed by Braverman et al. (Empty-car Routing in Ridesharing Systems) and utilizes NYC taxi data to demonstrate the results.


#### Programming language and platform:
- Python
- JupyterNotebook
#### Required packages:
- Numpy
- Matplotlib
- Networkx
#### File structure:
- nyc_trip
    |> Constructing NYC Taxi Grid -> notebook to visualize relocation in the NYC map 
    |> Learning Model Params      -> notebook to learn parameters of the relocation algo from NYC taxi data
    |> Solve Relocation Matrix    -> *core* notebook to solve relocation matrix, we can mod constraints here
    |> Qs.npz                     -> solved relocation matrix for discrete-time lookahead policy (in package)
- simulate
    |> simulate                   -> notebook to run simulation and evaluate the relocation matrix's effectiveness
    |> simulate.py                -> classes and objects for running the simulation 

