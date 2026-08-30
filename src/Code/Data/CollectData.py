from Code.Hardware.ClawMachine import ClawMachine
from Code.Vision.ReadCube import CubeReader

robot = ClawMachine()
reader = CubeReader(robot)

reader.ScanManyFacesData()