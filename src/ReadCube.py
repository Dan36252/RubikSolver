
# Procedure:
# Simply have camera read every face (order doesn't matter) using computer vision.
# Then, take the list of unordered faces data and order them according to Face Order (R, B, O, ...)
# Finally, feed this into CubeState, which will be able to flatten it and perform "data cube turns."

def ReadCube():
    print("~~~")