# Parameter Tuning Guide
To tune the parameters of the improved DBSCAN solution, here's a guide:

Start with Room Segmentation Parameters:

room_eps: Controls how far apart points can be to be considered part of the same room

Too large: Separate rooms might be merged
Too small: Single room might be split into multiple segments
Recommendation: Start with 10x your object eps value (e.g., 1.0 if object eps is 0.1)


room_min_samples: Minimum points to form a room cluster

Too large: Small rooms might be missed
Too small: Noise might be detected as rooms
Recommendation: Start with 2x your object min_points value




Tune Plane Extraction Parameters:

plane_distance: Maximum distance of a point to a plane to be considered part of it

Too large: May include non-planar points
Too small: May miss parts of walls/floors
Recommendation: Start around 0.05


min_plane_size: Minimum number of points to consider as a plane

Too large: Small walls/partitions might be missed
Too small: Might detect small flat objects as planes
Recommendation: Start around 500




Fine-tune Object Detection Parameters:

object_eps: Similar to your original DBSCAN eps parameter

Too large: Objects close together might be merged
Too small: Single object might be split into parts
Recommendation: Start with your original DBSCAN eps value


object_min_samples: Similar to your original DBSCAN min_points parameter

Too large: Small objects might be missed
Too small: Noise might be detected as objects
Recommendation: Start with your original DBSCAN min_points value
