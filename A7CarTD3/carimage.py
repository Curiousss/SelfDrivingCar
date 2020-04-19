import numpy as np
from PIL import Image as PILImage

def ArrowImage(car_pos, car_angle, car_size, map_size):
    map_np = np.ones(map_size) #(548, 975)
    map_img = PILImage.fromarray(map_np.astype("uint8")*255)
    map_img.save('blank.png')
    arrow_img = PILImage.open('arrow.png')
    arrow_img = arrow_img.rotate(car_angle)
    arrow_img = arrow_img.resize(car_size)
    map_img.paste(arrow_img, car_pos, arrow_img)
    return map_img
