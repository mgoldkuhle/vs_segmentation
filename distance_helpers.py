import numpy as np
# import timeit
from scipy.spatial.distance import pdist, squareform

# # dummy data for testing
# x_extra = np.random.rand(2000, 3)

# ## diameters any direction (exhaustive search). not recommended, very slow. just included for comparison.
# def exhaustive_search(x_extra):
#     max_distance = 0
#     for i in range(len(x_extra)):
#         for j in range(i + 1, len(x_extra)):
#             # Calculate the Euclidean distance between points[i] and points[j]
#             p1 = x_extra[i]
#             p2 = x_extra[j]
#             distance = np.linalg.norm(x_extra[i] - x_extra[j])
#             # If this distance is greater than the current max_distance, update max_distance
#             if distance > max_distance:
#                 max_distance = distance
#                 max_points = (x_extra[i], x_extra[j])
#     print(max_distance, max_points)
#     return max_distance, max_points


## just calculating the max distance, not returning points
def pdist_func(x_extra):
    distances = pdist(x_extra)
    max_distance = np.max(distances)
    return max_distance


# ## returning max distance and points, slightly slower but still way faster than exhaustive search
# def pdist_withpoints(x_extra):
#     distances = pdist(x_extra)
#     square_distances = squareform(distances)
#     max_distance = np.max(distances)
#     i, j = np.unravel_index(square_distances.argmax(), square_distances.shape)
#     max_points = (x_extra[i], x_extra[j])
#     print(max_distance, max_points)
#     return max_distance, max_points


# time the functions
# time_exhaustive = timeit.timeit('exhaustive_search(x_extra)', globals=globals(), number=10)

# time_pdist = timeit.timeit('pdist_func(x_extra)', globals=globals(), number=10)  # roughly 700x faster than exhaustive search

# time_pdist_indeces = timeit.timeit('pdist_withpoints(x_extra)', globals=globals(), number=10)  # roughly 200x faster than exhaustive search
