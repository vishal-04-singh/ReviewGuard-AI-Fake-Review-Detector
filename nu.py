height = [4,3,2,1,4]
b = sorted(list(height))

if len(b) == 1:
    print (b[0] * b[0])

c = b[-1]
d = b[-2]

print (c)
print (d)

if c < d:
    print (c*c)
elif c > d:
    print (d*d)
else:
    print (c*d)    



