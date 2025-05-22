from hdlib.space import Vector
import numpy as np

cv1 = Vector(vtype="bipolar", size=10000)
cv2 = Vector(vtype="bipolar", size=10000)

sample_vector = Vector(vtype="bipolar", size=10000)

"""
sample vector belongs to cv1 but the algorithm mistakenly says that it belongs to
cv2.

We need to increase the sample vector's presence in the correct class vector (via
addition) and decrease its presence in the incorrect class vector (using subtraction)

"""


cv1_arr = np.add(cv1.vector, sample_vector.vector)
cv1 = Vector(vector=cv1_arr)
cv1.normalize()

cv2_arr = np.subtract(cv2.vector, sample_vector.vector)
cv2 = Vector(vector=cv2_arr)
cv2.normalize()

print(cv1)
print(cv2)


