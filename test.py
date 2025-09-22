from scipy.integrate import trapezoid

y = [0,1,4,9,16]
x = [0,1,2,3,4]
s = trapezoid(y,x)
print(s)