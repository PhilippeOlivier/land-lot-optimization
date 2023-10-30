"""Land lot optimization.

We have a lot of size 60×40. We want to place up to 5 residential buildings (blue), up to 2 parking
lots (grey), and 1 park (green). There are two problematic areas on the lot. The first one (red) is
a floodable area, and nothing can be built there. The second one (orange) is a utility pole: we
cannot build a residential building there nor have it in the park, but we are allowed to build a
parking lot around it. The size of the park must be at least as large as the largest residential
building. The combined area of the parking lots must be at least 10% of the combined area of the
residential buildings. We want to generate a 2D plan that maximizes the yield of this lot, which is
the combined area of the residential buildings.
"""


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ortools.sat.python import cp_model


SIZE_X = 60
SIZE_Y = 40
NUM_BUILDINGS = 5
NUM_PARKING_LOTS = 2

FLOODABLE_X_START = 10
FLOODABLE_X_SIZE = 7
FLOODABLE_X_END = FLOODABLE_X_START + FLOODABLE_X_SIZE
FLOODABLE_Y_START = 20
FLOODABLE_Y_SIZE = 12
FLOODABLE_Y_END = FLOODABLE_Y_START + FLOODABLE_Y_SIZE

UTILITY_X_START = 40
UTILITY_X_SIZE = 5
UTILITY_X_END = UTILITY_X_START + UTILITY_X_SIZE
UTILITY_Y_START = 30
UTILITY_Y_SIZE = 5
UTILITY_Y_END = UTILITY_Y_START + UTILITY_Y_SIZE

model = cp_model.CpModel()


def entity_2d(model, name):
    """Return a 2D entity.

    {"X": {"Start": [X starting position],
           "Size": [length of X],
           "End": [X ending position],
           "Interval": [interval variable associated with Start, Size, and End]},
     "Y": {"Start": [Y starting position],
           "Size": [length of Y],
           "End": [Y ending position],
           "Interval": [interval variable associated with Start, Size, and End]},
    "Area": [area occupied by the entity]}
    """
    entity = {"X": {"Start": model.NewIntVar(0, SIZE_X, f"{name}_x_start"),
                    "Size": model.NewIntVar(0, SIZE_X, f"{name}_x_size"),
                    "End": model.NewIntVar(0, SIZE_X, f"{name}_x_end")},
              "Y": {"Start": model.NewIntVar(0, SIZE_Y, f"{name}_y_start"),
                    "Size": model.NewIntVar(0, SIZE_Y, f"{name}_y_duration"),
                    "End": model.NewIntVar(0, SIZE_Y, f"{name}_y_end")}}

    entity["X"]["Interval"] = model.NewIntervalVar(entity["X"]["Start"],
                                                   entity["X"]["Size"],
                                                   entity["X"]["End"],
                                                   f"{name}_x_interval")

    entity["Y"]["Interval"] = model.NewIntervalVar(entity["Y"]["Start"],
                                                   entity["Y"]["Size"],
                                                   entity["Y"]["End"],
                                                   f"{name}_y_interval")

    entity["Area"] = model.NewIntVar(0, SIZE_X*SIZE_Y, f"{name}_area")

    # Enforce the size of the area
    model.AddMultiplicationEquality(entity["Area"], [entity["X"]["Size"],
                                                     entity["Y"]["Size"]])

    return entity


# Building variables
buildings = {i: entity_2d(model, f"building_{i}")
             for i in range(NUM_BUILDINGS)}

# Symmetry breaking for buildings
for i in range(NUM_BUILDINGS-1):
    model.Add(buildings[i]["Area"] >= buildings[i+1]["Area"])

# Parking lots variables
parking_lots = {i: entity_2d(model, f"parking_lots_{i}")
                for i in range(NUM_PARKING_LOTS)}

# Symmetry breaking for parking lots
model.Add(parking_lots[0]["Area"] >= parking_lots[1]["Area"])

# Park variables
park = entity_2d(model, "park")

# Floodable interval variable
floodable_interval = {"X": model.NewIntervalVar(FLOODABLE_X_START,
                                                FLOODABLE_X_SIZE,
                                                FLOODABLE_X_END,
                                                "floodable_interval_x"),
                      "Y": model.NewIntervalVar(FLOODABLE_Y_START,
                                                FLOODABLE_Y_SIZE,
                                                FLOODABLE_Y_END,
                                                "floodable_interval_y")}

# Utility pole interval variable
utility_interval = {"X": model.NewIntervalVar(UTILITY_X_START,
                                              UTILITY_X_SIZE,
                                              UTILITY_X_END,
                                              "utility_interval_x"),
                    "Y": model.NewIntervalVar(UTILITY_Y_START,
                                              UTILITY_Y_SIZE,
                                              UTILITY_Y_END,
                                              "utility_interval_y")}

# The buildings, the parking lots, the park, and the floodable area cannot overlap
model.AddNoOverlap2D([buildings[i]["X"]["Interval"] for i in range(NUM_BUILDINGS)] +
                     [parking_lots[i]["X"]["Interval"] for i in range(NUM_PARKING_LOTS)] +
                     [park["X"]["Interval"]] +
                     [floodable_interval["X"]],
                     [buildings[i]["Y"]["Interval"] for i in range(NUM_BUILDINGS)] +
                     [parking_lots[i]["Y"]["Interval"] for i in range(NUM_PARKING_LOTS)] +
                     [park["Y"]["Interval"]] +
                     [floodable_interval["Y"]])

# The utility pole cannot overlap with the buildings
for i in range(NUM_BUILDINGS):
    model.AddNoOverlap2D([buildings[i]["X"]["Interval"], utility_interval["X"]],
                         [buildings[i]["Y"]["Interval"], utility_interval["Y"]])

# The utility pole cannot overlap with the park
model.AddNoOverlap2D([park["X"]["Interval"], utility_interval["X"]],
                     [park["Y"]["Interval"], utility_interval["Y"]])

# The combined areas of the parkings must be at least 10% the combined areas of the buildings
model.Add(cp_model.LinearExpr.Sum([parking_lots[i]["Area"] for i in range(NUM_PARKING_LOTS)]) * 10 >=
          cp_model.LinearExpr.Sum([buildings[i]["Area"] for i in range(NUM_BUILDINGS)]))

# The area of the park must be at least as large as the area of the largest building
largest_building = model.NewIntVar(0, SIZE_X*SIZE_Y, "largest_building")
model.AddMaxEquality(largest_building, [buildings[i]["Area"] for i in range(NUM_BUILDINGS)])
model.Add(park["Area"] >= largest_building)

# The objective is to maximize the yield (building area)
lot_yield = model.NewIntVar(0, SIZE_X*SIZE_Y, "lot_yield")
model.Add(lot_yield == cp_model.LinearExpr.Sum([buildings[i]["Area"] for i in range(NUM_BUILDINGS)]))
model.Maximize(lot_yield)

# Solve the problem with a time limit
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 5
status = solver.Solve(model)
if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("Model neither optimal nor feasible")

print(f"Yield: {solver.Value(lot_yield)}")

# Display results
fig, ax = plt.subplots()
plt.xlim([0, SIZE_X])
plt.ylim([0, SIZE_Y])

# Residential buildings
for i in range(NUM_BUILDINGS):
    ax.add_patch(Rectangle((solver.Value(buildings[i]["X"]["Start"]),
                            solver.Value(buildings[i]["Y"]["Start"])),
                           solver.Value(buildings[i]["X"]["Size"]),
                           solver.Value(buildings[i]["Y"]["Size"]),
                           edgecolor='black',
                           facecolor='royalblue',
                           fill=True,
                           lw=3))

# Parking lots
for i in range(NUM_PARKING_LOTS):
    ax.add_patch(Rectangle((solver.Value(parking_lots[i]["X"]["Start"]),
                            solver.Value(parking_lots[i]["Y"]["Start"])),
                           solver.Value(parking_lots[i]["X"]["Size"]),
                           solver.Value(parking_lots[i]["Y"]["Size"]),
                           edgecolor='black',
                           facecolor='grey',
                           fill=True,
                           lw=3))

# Park
ax.add_patch(Rectangle((solver.Value(park["X"]["Start"]),
                        solver.Value(park["Y"]["Start"])),
                       solver.Value(park["X"]["Size"]),
                       solver.Value(park["Y"]["Size"]),
                       edgecolor='black',
                       facecolor='limegreen',
                       fill=True,
                       lw=3))

# Floodable area
ax.add_patch(Rectangle((FLOODABLE_X_START,
                        FLOODABLE_Y_START),
                       FLOODABLE_X_SIZE,
                       FLOODABLE_Y_SIZE,
                       edgecolor='black',
                       facecolor='red',
                       fill=True,
                       lw=3))

# Utility pole
ax.add_patch(Rectangle((UTILITY_X_START,
                        UTILITY_Y_START),
                       UTILITY_X_SIZE,
                       UTILITY_Y_SIZE,
                       edgecolor='black',
                       facecolor='orange',
                       fill=True,
                       lw=3))

plt.show()
