import asyncio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ooflow

@ooflow.Node
async def A(ctx: ooflow.Context):
    count = 0
    while True:
        count = count + 1
        msg = await ctx.fetch()
        await ctx.emit(f"{count} {msg} Hello")

@ooflow.Node
async def B(ctx: ooflow.Context):
    while True:
        msg = await ctx.fetch()
        await ctx.emit(f"{msg} World")

@ooflow.Node
async def C(ctx: ooflow.Context):
    while True:
        msg = await ctx.fetch()
        await ctx.emit(f"{msg} !")

async def main():
    # Create and run the flow
    flow = ooflow.create(
        A.to(B),
        B.to(C)
    )

    flow.run()
    count_down = 3
    while count_down > 0:
        count_down = count_down - 1

        await flow.emit("start")
        print(await flow.fetch())
        await asyncio.sleep(1)
    flow.stop()

if __name__ == "__main__":
    asyncio.run(main())
