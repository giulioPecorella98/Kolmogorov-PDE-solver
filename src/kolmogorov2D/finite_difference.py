def forward_difference(u, h, i):
    du = (u[i+1] - u[i]) / h
    return du

def backward_difference(u, h, i):
    du = (u[i] - u[i-1]) / h
    return du

def central_difference(u, h, i):
    du = (u[i+1] - u[i-1]) / (2 * h)
    return du

def second_order_difference(u, h, i):
    d2u = (u[i+1] - 2 * u[i] + u[i-1]) / h**2
    return d2u
