# Land Lot Optimization

Optimize land lot yield using constraint programming.

The toy problem for this example is the following:

*We have a lot of size 60×40. We want to place up to 5 residential buildings (blue), up to 2 parking lots (grey), and 1 park (green). There are two problematic areas on the lot. The first one (red) is a floodable area, and nothing can be built there. The second one (orange) is a utility pole: we cannot build a residential building there nor have it in the park, but we are allowed to build a parking lot around it. The size of the park must be at least as large as the largest residential building. The combined area of the parking lots must be at least 10% of the combined area of the residential buildings. We want to generate a 2D plan that maximizes the yield of this lot, which is the combined area of the residential buildings.*

Read the related blog post [here](https://pedtsr.ca/2023/land-lot-optimization.html).
