from random import *
from  tkinter import *
nums = []

def roll_dice(sides, rolls):
    for i in range(rolls):
        nums.append(randint(1,sides))
    print(nums)

roll_dice(6,4)