from VisionPiece import Piece

class CubeExtractor:
    def extract_pieces(self, colors: list[list[int]], probabilities=None):
        extraction_indices = {
            "corners": [[[1, 6], [2, 0], [0, 2]],
                        [[1, 8], [4, 0], [2, 2]],
                        [[5, 0], [4, 2], [1, 2]],
                        [[1, 0], [0, 0], [5, 2]],
                        [[3, 0], [0, 8], [2, 6]],
                        [[3, 2], [2, 8], [4, 6]],
                        [[3, 8], [4, 8], [5, 6]],
                        [[3, 6], [5, 8], [0, 6]]],
            "edges": [[[0, 1], [1, 3]],
                      [[1, 7], [2, 1]],
                      [[1, 5], [4, 1]],
                      [[1, 1], [5, 1]],
                      [[0, 5], [2, 3]],
                      [[2, 5], [4, 3]],
                      [[4, 5], [5, 3]],
                      [[5, 5], [0, 3]],
                      [[0, 7], [3, 3]],
                      [[2, 7], [3, 1]],
                      [[4, 7], [3, 5]],
                      [[5, 7], [3, 7]]],
            "centers": [[0, 4], [1, 4], [2, 4], [3, 4], [4, 4], [5, 4]]
        }

        do_probs = (not probabilities is None)

        #print("Extracting corners...")
        corners = []
        for corner in extraction_indices["corners"]:
            corner_colors = []
            corner_indices = corner.copy()
            corner_probabilities = []
            for c in corner:
                corner_colors.append(colors[c[0]][c[1]])
                if do_probs: corner_probabilities.append(probabilities[c[0]][c[1]].copy())
            corners.append(Piece(corner_colors, corner_indices, corner_probabilities))

        #print("Extracting edges...")
        edges = []
        for edge in extraction_indices["edges"]:
            edge_colors = []
            edge_indices = edge.copy()
            edge_probabilities = []
            for e in edge:
                edge_colors.append(colors[e[0]][e[1]])
                if do_probs: edge_probabilities.append(probabilities[e[0]][e[1]].copy())
            edges.append(Piece(edge_colors, edge_indices, edge_probabilities))

        #print("Extracting centers...")
        centers = []
        for center in extraction_indices["centers"]:
            center_colors = [colors[center[0]][center[1]]]
            center_indices = [center.copy()]
            center_probabilities = []
            if do_probs: center_probabilities = [probabilities[center[0]][center[1]]]
            centers.append(Piece(center_colors, center_indices, center_probabilities))

        return corners, edges, centers