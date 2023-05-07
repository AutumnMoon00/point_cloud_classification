About file format: 
    Each 'xyz' file contains the point cloud of a single object, in which each line has three floating point numbers denoting the x, y, and z coordinates of a 3D point.

Ground truth labels:
    000 - 099: building
    100 - 199: car
    200 - 299: fence
    300 - 399: pole
    400 - 499: tree

Credit and references:
    All point clouds are taken from the DALES Objects dataset. More details about the dataset can be found in the following paper:
    Singer et al. DALES Objects: A Large Scale Benchmark Dataset for Instance Segmentation in Aerial Lidar, 2021.
