import numpy as np

print("--------------------simple backprop--------------------")

x = -2
y = 5
z = -4

print("x : {}".format(x), "y : {}".format(y), "z : {}".format(z))

print()

# forward pass q = 3, f = -12
q = x + y
f = q * z

print()

dfdz = q # df/dz = q, so gradient on z 3
dfdq = z # df/dq = z, so gradient on q -4
dqdx = 1.0 # dq/dx = 1
dqdy = 1.0 # dq/dy = 1

# chain rule
dfdx = dfdq * dqdx
dfdy = dfdq * dqdy

print("df/dx : {}".format(dfdx))
print("df/dy : {}".format(dfdy))

print("--------------------sigmoid backprop--------------------")

w = [2, -3, -3]
x = [-1, -2]

# forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]
f = 1.0 / (1 + np.exp(-dot))

print("dot : {}, f : {}".format(dot, f))

# backward pass
ddot = (1 - f) * f
dx = [w[0] * ddot, w[1] * ddot]
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot]

print("dx : {}, dw : {}".format(dx, dw))