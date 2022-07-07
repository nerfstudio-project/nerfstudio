# https://gist.github.com/jeffbass/ebf877e964c9a0b84272

# zmqimage.py -- classes to send, receive and display cv2 images via zmq
#     based on serialization in pyzmq docs and pyzmq/examples/serialization

"""
PURPOSE:
    These classes allow a headless (no display) computer running OpenCV code
    to display OpenCV images on another computer with a display.
    For example, a headless Raspberry Pi with no display can run OpenCV code
    and can display OpenCV images on a Mac with a display.
USAGE:
    First, start this "display server" program on the computer with a display:
        # imageShowServer.py 
        import zmqimage
        zmq = zmqimage.zmqImageShowServer()
        print "Starting zmqImageShow Server..."
        print "  press Ctrl-C to stop"
        while True:       # Until Ctrl-C is pressed, will repeatedly
            zmq.imshow()  # display images sent from the headless computer
    Run the above program by:
        python imageShowServer.py
    Leave the above program running in its own terminal window.
    Then, run a program like the one below on the headless computer. 
    In most cases, it will be run using ssh into the headless computer
    from another terminal window on the computer with a display.
    The python lines below represent a program fragment as an example.
    Use zmq.imshow("Image Display Name", image) instead of 
    cv2.imshow("Image Display Name", image) and the images will 
    display on the computer running the program above:
        import numpy as np
        import cv2
        import zmqimage
        print "Connecting to zmqShowImage Server ... "
        zmq = zmqimage.zmqConnect()
        image = np.zeros((500, 500), dtype="uint8")
        zmq.imshow("Zero Image 500 x 500", image)
        # build a rectangular mask & display it
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.rectangle(mask, (0, 90), (300, 450), 255, -1)
        zmq.imshow("Rectangular Mask", mask)
CAVEATS:
    There is no error checking and only Ctrl-C stops the display server.
    While zmq.imshow() works like cv2.imshow(), no other 
    cv2 display functions are implemented.
    Uses tcp style sockets; sockets and tcp addresses have
    defaults in the classes but may be overridden.
AUTHOR:
    Jeff Bass, https://github.com/jeffbass, jeff@yin-yang-ranch.com
"""

import zmq
import numpy as np
import cv2


class SerializingSocket(zmq.Socket):
    """A class with some extra serialization methods

    send_array sends numpy arrays with metadata necessary
    for reconstructing the array on the other side (dtype,shape).
    Also sends array name for display with cv2.show(image).
    recv_array receives dict(arrayname,dtype,shape) and an array
    and reconstructs the array with the correct shape and array name.
    """

    def send_array(self, A, arrayname="NoName", flags=0, copy=True, track=False):
        """send a numpy array with metadata and array name"""
        md = dict(
            arrayname=arrayname,
            dtype=str(A.dtype),
            shape=A.shape,
        )
        self.send_json(md, flags | zmq.SNDMORE)
        return self.send(A, flags, copy=copy, track=track)

    def recv_array(self, flags=0, copy=True, track=False):
        """recv a numpy array, including arrayname, dtype and shape"""
        md = self.recv_json(flags=flags)
        msg = self.recv(flags=flags, copy=copy, track=track)
        A = np.frombuffer(msg, dtype=md["dtype"])
        return (md["arrayname"], A.reshape(md["shape"]))


class SerializingContext(zmq.Context):
    _socket_class = SerializingSocket


class zmqConnect:
    """A class that opens a zmq REQ socket on the headless computer"""

    def __init__(self, connect_to="tcp://jeff-mac:5555"):
        """initialize zmq socket for sending images to display on remote computer"""
        """connect_to is the tcp address:port of the display computer"""
        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REQ)
        self.zmq_socket.connect(connect_to)

    def imshow(self, arrayname, array):
        """send image to display on remote server"""
        if array.flags["C_CONTIGUOUS"]:
            # if array is already contiguous in memory just send it
            self.zmq_socket.send_array(array, arrayname, copy=False)
        else:
            # else make it contiguous before sending
            array = np.ascontiguousarray(array)
            self.zmq_socket.send_array(array, arrayname, copy=False)
        message = self.zmq_socket.recv()


class zmqImageShowServer:
    """A class that opens a zmq REP socket on the display computer to receive images"""

    def __init__(self, open_port="tcp://*:5555"):
        """initialize zmq socket on viewing computer that will display images"""
        self.zmq_context = SerializingContext()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(open_port)

    def imshow(self, copy=False):
        """receive and show image on viewing computer display"""
        arrayname, image = self.zmq_socket.recv_array(copy=False)
        # print "Received Array Named: ", arrayname
        # print "Array size: ", image.shape
        cv2.imshow(arrayname, image)
        cv2.waitKey(0)
        self.zmq_socket.send(b"OK")
