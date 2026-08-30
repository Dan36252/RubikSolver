## Summary
<img src=photos/FullRobotPhoto.jpg alt="Rubiks Cube Robot" width=700>
Hello! This is the code, datasets, and model weights for my Rubik's Cube solving robot.

The robot works like so:
* You place a scrambled cube in a specific orientation
* Robot turns the cube so one face is towards the camera
* Robot recognizes the colors on that face using a neural network
* The rest of the faces are pointed at the camera and read
* Once the robot knows the Rubik's cube configuration, it can start solving
* For solving the First Two Layers (F2L) of the cube, the robot uses heuristic search to find the best combination of moves
* For solving the rest of the cube, the robot uses a neural network trained to perform known formulas when given a cube state

<br><br>
## Building Process


| I came up with this idea because I wanted to make a robot that uses AI to do something cool.
| ---
| <br> First, I sketched some basic mechanisms and brainstormed how the robot would actually work. The robot can turn any side of the cube using only four claws and a platform at the bottom, because the claws can rotate the entire cube to then turn the top and bottom faces. To turn one side of the cube, the robot needs to hold the cube so no other layers rotate.
| <img src=photos/FirstSketches.jpg alt="First Sketches" width=400 align="left">
| <br> I then designed the necessary parts in OnShape and 3D printed them. The claws have two servo motors each - one for extending the arm (to latch onto/release cube faces) and one for rotating the face.
| <img src=photos/CAD3.png alt="Designing Parts" height=200 align="left"> <img src=photos/3DPrintedClaw.jpg alt="3D Printing Claws" height=200 align="left">
| <br> Next, I learned how to control the servo motors using an NVIDIA Jetson Nano (the computer that runs the program) and a special component called a PCA9685 (which allows you to control up to 16 servos/LEDs using just 2-4 wires).
| <img src=photos/HardwareSetup.jpg alt="Hardware Setup" height=200 align="left">
| <br> Finally, I assembled the robot. I loaded the claws with springs to make the claws less prone to getting stuck. Also, I played around with different materials for connecting the arm extendor servos to the arms themselves. The robot frame was built out of LEGO robotics pieces, which made it easy to prototype and iterate on the design.
| <img src=photos/ClawAssembly.jpg alt="Claw Assembly" height=200 align="left"> <img src=photos/FrameAssembly.jpg alt="Frame Assembly" height=200 align="left"> <img src=photos/FullAssembly.jpg alt="Full Assembly" height=200 align="left">

<br><br><br>
## Coding Process


| To code the robot, I used Python and trained a few custom machine learning models using PyTorch and a Kaggle dataset.
| ---
| <br> First, I worked on the main algorithm: the one that actually solves a Rubik's cube. My idea was, if a computer could learn what move is best to make when it sees a specific Rubik's cube configuration, it would be able to solve the entire cube. <br><br> So, I found a dataset on Kaggle that has just what I needed: tens of thousands of Rubik's Cube configurations, each with a corresponding "next move" that bring the cube closer to a solved state. Here is the dataset: [Kaggle.com](https://www.kaggle.com/datasets/antbob/rubiks-cube-cfop-solutions) (Thank you, Anton) <br><br> With this dataset, I trained a dense neural network using PyTorch, which takes 54 numbers as an input (representing a rubik's cube configuration) and outputs 19 numbers, one for each type of move you can make (including "stop").
| <img src=photos/ModelTraining1.jpg alt="Main Model Training" height=300 align="left">
| <br> However, although the model claimed to have 100% accuracy, this was not so. When I ran the model on many scrambled Rubik's cubes, it only got 30-50% of the moves correct, and for some reason the last two moves before the cube was solved were always predicted wrong by the model. <br><br> I asked ChatGPT and my dad for suggestions, and I tried to include the previous 30 moves the robot already did as an input to the model, and also made the model predict how many moves away from being solved the input cube state is. But these only helped a little. <br><br> So, I moved on to work on other parts of the code. I will revisit this model later.
| <br> Next, I worked on making the robot actually turn and manipulate the Rubik's cube. In Python, I created a "Claw" class, which allows me to control each individual claw: extend/retract, twist, set_angle, etc. I also created a "Claw Machine" class, which is in charge of orchestrating all four claw movements and actually manipulating the cube. Some key methods in this class are "turn_face()," which turns any given face one time clockwise; "turn_cube()," which rotates the entire cube around a given axis; and "move()," which combines the previous two methods to turn any face in any direction. <br><br> Here is a demo of the robot turning the Rubik's cube:

https://github.com/user-attachments/assets/ab9e30f1-9ed8-4a56-907b-b8ae0959ce43

| To make the robot "see" what colors the Rubik's cube has on each face, I mounted a camera, connected it to the Jetson Nano, and created a few versions of a program that extracts the colors of each face.
| ---
| <br> My first idea was to hard-code a program that took an image of a cube face, divided it into a 3x3 grid, took the average color of each section, and compared it to a list of known colors. Whichever color each section was closest to, that was the predicted color for that sticker. This approach worked surprisingly well; but it was still not 100% accurate. <br><br> So, I decided to try something else: detecting the colors using a neural network. To do this, I first needed to find a large dataset of Rubik's cube face images, each labeled with the correct color pattern. I could not find a good dataset, so I did what my dad suggested. I used the robot I already had to scramble, turn, and take pictures of the cube. Using a custom index remapping dictionary, the computer keeps track of the Rubik's cube configuration after each rotation, so the robot could automatically label each of the pictures it took with the correct color pattern. <br><br> Here is a timelapse of the automatic data collection process:

https://github.com/user-attachments/assets/1456aa78-bdb7-439f-9755-5622f268d804

| After this, I trained the Vision neural network.
| ---
| <br> Using only dense layers in the network allowed me to reach 95% accuracy, but when testing on the real robot, the model performed rather poorly. For example, it detected a mix of three different colors, when the Rubik's cube face was only red stickers.
| <img src=photos/VisionExample2.jpg alt="Actual colors" height=200 align="left"> <img src=photos/VisionExample1.jpg alt="Computer prediction" height=200 align="left">
| <br> My next idea is to use a Convolutional Neural Network instead. It will be able to identify the edges/boundaries between the stickers, and hopefully it will be easier to classify sampled-down data into the 9 different colors using this network.

Currently, I am just short of finishing the project. I still need to train a more reliable Vision model, and need to implement the main Rubik's Cube solving algorithm with the improvements I mentioned: using heuristic search for F2L and a formula-based neural network for the rest of the puzzle.

By heuristic search, I mean this:
1. Training a neural network to take an input cube configuration, and predict how "good" that configuration is - that is, how many moves will it take to solve from this point.
2. Given a scrambled cube, the program will try every possible move combination (up to 4 moves in depth), and the move sequence with the best final configuration (the one that the above neural network predicts is closest to being solved) will be selected and those moves will be performed.
3. The process repeats until the First Two Layers of the Rubik's Cube are solved.

By "formula-based" neural network, I mean this:
1. Training a neural network to take an input cube configuration, and predict what specific formula is best to apply in this situation.
2. This network will be trained using the same dataset I used for the first version of the Cube Solving algorithm.

Finally, while the project is still work-in-progress, here is a demo video, where the robot "solves" a Rubik's cube by following a predetermined formula that I wrote:
VIDEO DEMO: [YouTube - Robot Following a Formula](https://youtu.be/MRaupFCNtyo)

Thank you, God Bless!


Update: Currently, I'm working on a "clean-up" algorithm for the Vision system. Its goal is to take the raw color predictions that the CNN got from the camera images, count how many of each color the model predicted there were, and adjust the color predictions to make sure there are exactly 9 of each color, by using the next-best color probabilities the model determined.
