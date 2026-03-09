
class Piece:
    def __init__(self, colors, indices=None, probabilities=None):
        # colors: a list of 3 ints
        self.colors = colors
        self.piece_type = len(colors)
        # indices: a list of 3 lists, each with 2 ints: the sticker's face, and position on the face.
        self.indices = indices
        # probabilities: a list of 3 lists, corresponding to self.colors. each list contains 9 floats: the logits of that sticker.
        self.probabilities = probabilities

        self.orientation = 0  # Keep track of the original orientation of the piece; update when __rotate() called

        #self.standardize()

    def __eq__(self, other):
        if type(other) != Piece:
            return False
        if other.piece_type != self.piece_type:
            return False
        for i in range(len(self.colors)):
            if self.colors[i] != other.colors[i]:
                return False
        return True

    def __rotate(self):
        # Cycle the colors and probabilities lists to represent rotating the piece

        new_colors = [-1]*self.piece_type
        new_probabilities = [None]*self.piece_type

        do_probs = False if self.probabilities is None else len(self.probabilities) > 0

        i = 1
        while i < self.piece_type:
            new_colors[i] = self.colors[i-1]
            if do_probs: new_probabilities[i] = self.probabilities[i-1]
            i += 1

        new_colors[0] = self.colors[self.piece_type-1]
        if do_probs: new_probabilities[0] = self.probabilities[self.piece_type-1]

        self.colors = new_colors
        if do_probs: self.probabilities = new_probabilities
        self.orientation = self.orientation - 1 if self.orientation > 0 else (self.piece_type - 1)

    def standardize(self):
        if self.piece_type == 1: return

        success = False
        for i in range(self.piece_type):
            self.__rotate()
            top_color = self.colors[0]
            if top_color == 1 or top_color == 3:
                success = True
                break

        if success == False and self.piece_type == 2:
            for i in range(self.piece_type):
                self.__rotate()
                top_color = self.colors[0]
                if top_color == 0 or top_color == 4:
                    success = True
                    break

        status = "failed" if success == False else "successful!"
        print(f"Piece standardization: {status}")

