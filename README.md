## Summary
<img src=photos/FullRobotPhoto.jpg alt="Rubiks Cube Robot" width=700>
Hello! This is the code for my Rubik's Cube solving robot. (Not all datasets and model weights are uploaded yet.)

The robot is intended to work like so:
* The robot scans each face of the cube with a camera (using a computer vision model)
* The computer reconstructs the cube configuration in memory
* The computer runs an ML model to obtain a set of moves
* The robot performs those moves to solve the cube

Progress timeline:
* 12/3/25 - Got idea to make AI Rubik's Cube robot
* 12/16/25 - Trained MCP solver model #1 (Accuracy: 60.3%, Solve Rate: 0.0%)
* 12/17/25 - Trained MCP solver model #2 (Accuracy: 83.0%, Solve Rate: 0.0%)
* 12/26/25 - Finished first robot prototype (CAD, 3D printing, connected servos to Jetson)
* 12/30/25 - Began manually collecting image data for CV model
* 1/4/26 - First automatic image data mass-collection for CV model
* 1/6/26 - Trained MCP computer vision model (not accurate enough to reconstruct cube)
* 2/27/26 - Trained CNN computer vision model (still not accurate enough)
* 6/7/26 - Trained LSTM solver model #1 (Accuracy: 94.8%, Solve Rate: 2.0%)  YAY!
* 7/12/26 - Trained LSTM solver model #2 with data masking (Accuracy: 97.0%, Solve Rate: 42.0%)
* 8/24/26 - Trained Transformer solver model #1 (Accuracy: 97.4%, Solve Rate: 0.0%)
* (Still working on robot!)

<br>

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
| <br> My next idea was to use a Convolutional Neural Network instead. I thought it would be able to identify the edges/boundaries between the stickers, and hopefully it will be easier to classify images whose resolution is sampled-down by the CNN. Although the model performed slightly better than the multi-layer peceptron (above), it was not accurate enough to scan and reconstruct the entire cube.
| <br> I then realized that the Rubik's cube has exactly 9 stickers of each of the 6 colors. If the vision model predicts that there are more or less than 9 stickers of any color, it should change its prediction to match this restriction. I first thought of implementing this by looping over each color the cube has, and if the predicted number of stickers of that color is above 9, the excess stickers will be changed to the next most-probable color (preferably to a color that has a deficit of predicted stickers). But then I realized that this algorithm might not result in a legal Rubik's cube too. On a real Rubik's cube, some stickers cannot appear adjacently to each other on the same cubie. So a new algorithm I thought of was taking the raw sticker color predictions of the CNN, extracting those sticker colors to create a set of "piece" objects (which correspond to actual cubie pieces on a cube, each piece containing 1 to 3 stickers), and then taking a separate set of actually legal cubie piece objects and matching the predicted pieces onto the legal pieces. In other words, the algorithm ensures that the scanned cube is legal by taking actual legal sticker combinations and projecting them onto the CNN's scanned stickers in the most probable way. This last algorithm was implemented with the help of the AI-assitant Cursor.
| <br> Currently, I plan to improve the vision model's performance by implementing a transformer-based CNN, with the idea that variations in the lighting, reflections, and color-warping captured by the camera can be decoded if the model pays attention to all the pixels in the image at once. Another idea for improving the vision model is capturing all six sides of the cube and training the model to classify all six sides at once, which might help it compare all the colors on the cube and take their relative lighting/color warp into account.

<br>

| Then, I attempted to improve the Cube Solving Model.
| ---
| <br> As school was ending, I had more time to read, so I became interested in the idea of using Recurrent Neural Networks to analyze the moves already performed on the cube and classify which next move is best. I also got some new ideas for improving the model - namely, dividing the task of solving the cube into solving the first two layers using one model, and the last layer using a separate specialized model; and also grouping common sets of moves in the dataset into "formula tokens" to reduce the number of predictions the model needs to make to reach the solution.
| <br> After understanding how RNNs work and are trained, I moved on to exploring a better type of RNN - an LSTM. Seeing how an LSTM has both long-term and short-term memory, I thought it would be perfect for analyzing a long chain of Rubik's Cube moves (that were already performed on the cube), and outputting an even better-informed prediction of the next move. With that, I attempted to design a custom 3-layer LSTM model that also feeds its output into all hidden cells. The performance was surprising! It had a 97.0% move prediction accuracy, and it solved 21/50 cubes from the evaluation dataset (42%)!
| <img src=photos/LSTMTrainResults.png alt="LSTM Model Training Results" height=400 align="left">
| <br> Then, after learning about transformers, I thought that self-attention among the cube states that were already seen in the duration of solving the cube would improve the model accuracy. I tried implementing a custom decoder-only transformer model, but although it had a move-prediction accuracy of 97.4%, it could not solve the first two layers of the cube in a reasonable number of moves.

<br>
Currently, the project is not finished. I still need to train a more reliable Vision model, and need to improve the main Rubik's Cube solving algorithm. Some improvements I want to try for the solver model is using heuristic search on top of the policy model to improve reliability, applying reinforcement learning to help the model learn to create patterns on the cube instead of just memorizing sequences of moves, and making the model predict how close it is to solving the cube, which could help the model understand its progress and what the correct move is better.
<br><br>
For now, while the project is still work-in-progress, here is a demo video, where the robot "solves" a Rubik's cube by following a predetermined formula that I wrote:
VIDEO DEMO: [YouTube - Robot Following a Formula](https://youtu.be/MRaupFCNtyo)

Thank you, God Bless!
