#19 factorial 
def factorial(n):
    if n==0:
        return 1
    else:
        return n*factorial(n-1)
    
print(factorial(19)) # 121645100408832000

tiempo  = 121645100408832000/1000000000/60/60/24/365
print(tiempo) # 3851.401917808219