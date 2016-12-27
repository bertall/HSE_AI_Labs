# AI in videogames project
## Report
#### Bot for Arkanoid game
Author: Baishev Albert, group 145

### Using technologies
Python 3.5
Unreal Engine 4
Tensorflow
OpenCV

### Gameplay
At start of game we have table with blocks and paddle. 
Bot must beat the ball and break the blocks. For each broken block the player gets points.
The player has a limited number of lives, for which he has to break all the blocks on the field

### Specifications
For training bot we used a method known as Q learning, which approximates the maximum expected return for performing an action at a given state using an action-value (Q) function.
For learning, I used the following system of rewards: the loss of life is given a negative reward -1, with the ball staying on the field more than 100 ticks award is given 0.01 for each subsequent tick.

### Results
After approximately 1.2 million time steps, which corresponds to about 65 hours of game time.
In comparison with the scripted bot use neural network improves the quality of the game. Our bot hits a lot more balls than scripted

### Links
5 lives
YouTube: https://youtu.be/-V7RFQgA138