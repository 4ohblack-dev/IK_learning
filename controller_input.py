import time
from dataclasses import dataclass
import pygame

def deadzone(v,threshold=0.05):
    return 0 if abs(v) < threshold else v
@dataclass
class Input:
    leftX : float=0.0
    leftY : float=0.0
    rightY : float=0.0
    rightX : float=0.0
    rotL : float=0.0
    rotR : float=0.0
    L_button : int = 0
    R_button : int = 0
    up_button : int = 0
    down_button : int = 0
    left_button : int = 0
    right_button : int = 0
    cross_button : int = 0
    circle_button : int = 0
    tri_button : int = 0
    rec_button : int = 0

@dataclass
class ControlData:
    d_leftX : float=0.0
    d_leftY : float=0.0
    d_rightY : float=0.0
    d_rightX : float=0.0
    d_rotL : float=0.0
    d_rotR : float=0.0
    d_L_button : int = 0
    d_R_button : int = 0
    d_up_button : int = 0
    d_down_button : int = 0
    d_left_button : int = 0
    d_right_button : int = 0
    d_cross_button : int = 0
    d_circle_button : int = 0
    d_tri_button : int = 0
    d_rec_button : int = 0
class Controller(object):

    def __init__(self):
        self.isConnect = False
        self.joy = None

    def getController(self)-> bool:
        self.isConnect = False
        pygame.init()
        pygame.joystick.init()
        count = pygame.joystick.get_count()
        if count == 0:
            return False

        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        self.isConnect=True
        return True

    def read(self)->Input:
        if self.joy==None or self.isConnect==False:
            raise ValueError("controller not found")

        for event in pygame.event.get():
            if event.type == pygame.JOYDEVICEREMOVED:
                if event.instance_id == self.joy.get_instance_id():
                    self.isConnect = False
                    raise ValueError("controller removed")

        return Input(
            leftX = deadzone(self.joy.get_axis(0)),
            leftY = deadzone(self.joy.get_axis(1)),
            rightX = deadzone(self.joy.get_axis(2)),
            rightY = deadzone(self.joy.get_axis(3)),
            rotL = deadzone(self.joy.get_axis(4)),
            rotR = deadzone(self.joy.get_axis(5)),
            L_button = self.joy.get_button(9),
            R_button = self.joy.get_button(10),
            up_button = self.joy.get_button(11),
            down_button = self.joy.get_button(12),
            left_button = self.joy.get_button(13),
            right_button = self.joy.get_button(14),
            cross_button = self.joy.get_button(0),
            circle_button = self.joy.get_button(1),
            tri_button = self.joy.get_button(3),
            rec_button = self.joy.get_button(2),        
        )

    def inputToDelta(self)->ControlData:
        input = self.read()

        return ControlData(
            d_leftX = 0.1*input.leftX,
            d_leftY = 0.1*input.leftY,
            d_rightX = 0.1*input.rightX,
            d_rightY = 0.1*input.rightY,
            d_rotL = 0.1*input.rotL,
            d_rotR = 0.1*input.rotR,
            d_L_button = input.L_button,
            d_R_button = input.R_button,
            d_up_button = input.up_button,
            d_down_button = input.down_button,
            d_left_button = input.left_button,
            d_right_button = input.right_button,
            d_cross_button = input.cross_button,
            d_circle_button = input.circle_button,
            d_tri_button = input.tri_button,
            d_rec_button = input.rec_button,     
        )
    def close(self):
        if self.joy is not None:
            self.joy.quit()
            self.joy = None

        self.isConnect = False
        pygame.quit()

        
def main():
    controller = Controller()

    try:
        if not controller.getController():
            print("controller not found")
            return

        while True:
            inputs = controller.read()
            print(inputs)
            time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    except (ValueError, pygame.error) as error:
        print(error)
    finally:
        controller.close()

if __name__ == "__main__":
    main()
