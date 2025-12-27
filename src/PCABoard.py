from adafruit_pca9685 import PCA9685
import board

class PCABoard:
    single_object = None

    # Use this method to get the "singleton" PCA9685 object.
    def get(self):
        if PCABoard.single_object is None:
            i2c = board.I2C()
            PCABoard.single_object = PCA9685(i2c)
            PCABoard.single_object.frequency = 50
            return PCABoard.single_object
        else:
            return PCABoard.single_object