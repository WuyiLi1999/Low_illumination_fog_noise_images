import numpy as np
from PIL import Image

img = Image.open("output1.jpg").convert("RGB")
p=np.array(img)
x=p[0]
y=p[1]
z=p[2]
print(x)
print('---------')
print(y)
print('---------')
print(z)
print(img.size)