import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ooflow

@ooflow.Node
async def A(ctx: ooflow.Context):
    try:
        msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
        for i in range(3):
            await ctx.emit(f"A_{msg}_{i}")
    except asyncio.TimeoutError:
        pass

class BClass:
    @ooflow.Node
    async def B(self, ctx: ooflow.Context):
        while True:
            try:
                msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
                await ctx.emit(f"B_{msg}")
            except asyncio.TimeoutError:
                break
    # 等价于
    # B = ooflow.Node(B)

class CClass:
    @classmethod
    @ooflow.Node
    async def C(cls, ctx: ooflow.Context):
        while True:
            try:
                msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
                await ctx.emit(f"C_{msg}")
            except asyncio.TimeoutError:
                break
    # 等价于
    # C = classmethod(ooflow.Node(C))

class DClass:
    @staticmethod
    @ooflow.Node
    async def D(ctx: ooflow.Context):
        results = []
        while True:
            try:
                msg = await asyncio.wait_for(ctx.fetch(), timeout=0.1)
                results.append(f"D_{msg}")
                await ctx.emit(results)
            except asyncio.TimeoutError:
                break

    # 等价于
    # D = staticmethod(ooflow.Node(C))

print(f"CClass.C == CClass.C: {CClass.C == CClass.C}")
print(f"CClass.C is CClass.C: {CClass.C is CClass.C}")

print("All tests finished")
