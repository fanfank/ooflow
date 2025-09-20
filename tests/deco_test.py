"""
By fanfank: this script is to test the behavior of @classmethod and @deco
"""
import asyncio
import sys
import os

class deco:
    def __init__(self, func):
        print(f"deco.__init__: func={func}")
        self.func = func
        self.bound = None

    def __call__(self, *args, **kwargs):
        print(f"deco.__call__: args={args}, kwargs={kwargs}")
        if self.bound:
            return self.bound(*args, **kwargs)
        return self.func(*args, **kwargs)

    def __get__(self, instance, owner):
        print(f"deco.__get__: instance={instance}, owner={owner}")
        print(f"              type(instance)={type(instance)}")
        if self.bound:
            return self

        self.bound = self.func.__get__(instance, owner)
        return self

class A:
    # we can switch the positions of @classmethod and @deco
    #   to see the differences in the output
    @deco
    @classmethod
    def classfunc(cls):
        print("classfunc is called")

    @deco
    @staticmethod
    def staticfunc():
        print("staticfunc is called")

    @deco
    def instancefunc(self):
        print("instancefunc is called")

print("------ about to call A.classfunc() ------")
A.classfunc()
print("")

print("------ about to call a.classfunc() ------")
a = A()
a.classfunc()
print("")

print("------ about to call A.staticfunc() ------")
A.staticfunc()
print("")

print("------ about to call a.staticfunc() ------")
a.staticfunc()
print("")

print("------ about to call a.instancefunc() ------")
a.instancefunc()
print("")

print("++++++ about to do equal tests ++++++")
print("A.classfunc == A.classfunc:", A.classfunc == A.classfunc)
print("A.classfunc == a.classfunc:", A.classfunc == a.classfunc)
print("a.classfunc == a.classfunc:", a.classfunc == a.classfunc)
print("A.staticfunc == a.staticfunc:", A.staticfunc == a.staticfunc)
print("a.instancefunc == a.instancefunc:", a.instancefunc == a.instancefunc)
print("")

print("++++++ about to do 'is' tests ++++++")
print("A.classfunc is A.classfunc:", A.classfunc is A.classfunc)
print("A.classfunc is a.classfunc:", A.classfunc is a.classfunc)
print("a.classfunc is a.classfunc:", a.classfunc is a.classfunc)
print("A.staticfunc is a.staticfunc:", A.staticfunc is a.staticfunc)
print("a.instancefunc is a.instancefunc:", a.instancefunc is a.instancefunc)
print("")

print("------ All tests finished ------")