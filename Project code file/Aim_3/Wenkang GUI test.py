#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/11/15 13:57
# @Author  : Yiyang
# @File    : Wenkang GUI test.py
# @Contact: jy522@ic.ac.uk

from tkinter import *
from tkinter.ttk import Combobox

window = Tk()
var = StringVar()
var.set("one")
data = ("one", "two", "three", "four")
cb = Combobox(window, values=data)
cb.place(x=60, y=150)

lb = Listbox(window, height=5, selectmode='multiple')
for num in data:
    lb.insert(END, num)
lb.place(x=250, y=150)

v0 = IntVar()
v0.set(1)
r1 = Radiobutton(window, text="male", variable=v0, value=1)
r2 = Radiobutton(window, text="female", variable=v0, value=2)
r1.place(x=100, y=50)
r2.place(x=180, y=50)

v1 = IntVar()
v2 = IntVar()
C1 = Checkbutton(window, text="Cricket", variable=v1)
C2 = Checkbutton(window, text="Tennis", variable=v2)
C1.place(x=100, y=100)
C2.place(x=180, y=100)
print("Value of IntVar()", r1.wait_variable)
window.title('Hello Python')
window.geometry("400x300+10+10")
window.mainloop()