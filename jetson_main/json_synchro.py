import websockets
import asyncio
import json

async def checkMode(websocket):
    messageRecu = await websocket.recv()
    messageEnvoie = "Changement reçu"
    with open('data.json') as fp:
        mode = json.load(fp)
    if messageRecu == "manual":
        mode.update({"Control Mode" : "manual"})
        #mode["Control Mode"] = "manual"
    elif messageRecu == "autonomous":
        mode.update({"Control Mode": "autonomous"})
        #mode["Control Mode"] = "autonomous"
    elif messageRecu == "Park mode : on":
        mode.update({"Park Mode": "on"})
        #mode["Park Mode"] = "on"
    elif messageRecu == "Park mode : off":
        mode.update({"Park Mode": "off"})
        #mode["Park Mode"] = "off"
    with open('data.json','w') as fp:
        json.dump(mode,fp)
    print(mode)
    await websocket.send(messageEnvoie)

mode = {"Control Mode":"autonomous","Park Mode":"off"}
with open('data.json','w') as fp:
    json.dump(mode,fp)
    fp.close()

ipServeur = '10.30.50.185'
connexionServeur = websockets.serve(checkMode,ipServeur, 5679)
asyncio.get_event_loop().run_until_complete(connexionServeur)
asyncio.get_event_loop().run_forever()
