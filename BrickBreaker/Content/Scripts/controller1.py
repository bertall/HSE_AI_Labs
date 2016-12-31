import sys
import random

import unreal_engine as ue

ue.log("Python version: ".format(sys.version))

import numpy as np
import cv2

H = 240
W = 240


class PythonAIController(object):

    # Called at the started of the game
    def begin_play(self):
        ue.log("Begin Play on PythonAIController class")
        
    def get_screen(self, game_mode):
        if not game_mode:
            return None
        screenshot = np.array(game_mode.ScreenCapturer.Screenshot)

        if len(screenshot) == 0:
            return None

        return screenshot.reshape((W, H, 3), order='F').swapaxes(0, 1)

    # Called periodically during the game
    def tick(self, delta_seconds : float):
        pawn = self.uobject.GetPawn() 
        # ball_position = self.get_ball_position(pawn.GameMode) 
        game_state = pawn.GameState.Score 
        ue.log("Score: {}".format(pawn.GameState.score)) 
        pawn.action = random.randint(0, 2)
        
        screen = self.get_screen(pawn)
        if not screen is None:
            ue.log("Screen shape: {}".format(screen.shape))
            cv2.imwrite("/tmp/screen.png", 255.0 * screen)
        else:
            ue.log("Screen is not available") 
