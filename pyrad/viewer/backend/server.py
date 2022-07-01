from aiohttp import web
import socketio

ROOM = "room"


app = web.Application()
# app.add_routes([web.static("/web", "public/")])

sio = socketio.AsyncServer(cors_allowed_origins="*")
sio.attach(app)


@sio.event
async def connect(sid, environ):
    print("Connected", sid)
    await sio.emit("ready", room=ROOM, skip_sid=sid)
    sio.enter_room(sid, ROOM)


@sio.event
def disconnect(sid):
    sio.leave_room(sid, ROOM)
    print("Disconnected", sid)


@sio.event
async def data(sid, data):
    print("Message from {}: {}".format(sid, data))
    await sio.emit("data", data, room=ROOM, skip_sid=sid)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8052)

    args = parser.parse_args()

    web.run_app(app, host=args.host, port=args.port)
