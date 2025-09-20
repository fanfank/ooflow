import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import ooflow

@ooflow.Node
async def A(context: ooflow.Context):
    msg = await context.fetch()
    await context.emit(msg + "A | ")

@ooflow.Node
async def B(context: ooflow.Context):
    msg = await context.fetch()
    await context.emit(msg + "B | ", C)
    await context.emit(msg + "B | ", D)

@ooflow.Node
async def C(context: ooflow.Context):
    msg = await context.fetch()
    await context.emit(msg + "C | ")

@ooflow.Node
async def D(context: ooflow.Context):
    msg_B = await context.fetch(B)
    msg_C = await context.fetch(C)

    await context.emit((msg_B + "D | ", msg_C + "D | "))

@ooflow.Node
async def E(context: ooflow.Context):
    await context.emit(await context.fetch())

async def myfunc():
    flow_instance = ooflow.create(
        A.to(B),
        B.to(C, D),
        C.to(D),
        D.to(E)
    )

    flow_instance.run()
    await flow_instance.emit("ooflow by fanfank ")

    result = await flow_instance.fetch()
    print(result)

if __name__ == "__main__":
    asyncio.run(myfunc())
