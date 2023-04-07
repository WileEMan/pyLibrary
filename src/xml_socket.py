""" Simple XML Communications Socket

    Copyright (C) 2021-2022 by Wiley Black

    Provides a simple socket-based XML messaging protocol for client and server implementation.  The protocol
    prefixes each messages with a 4-byte message length, then the number of bytes specified.  Each message is 
    expected to be in an XML format, but the details of content are left to the caller.  
    
    An optional <Disconnect/> message can automatically be sent by the client to indicate that it is about to 
    disconnect.  This may help with diagnostics by showing that the tunnel was closed intentionally by the
    client.

    The Server class can be used to implement the server-side, including a listener socket monitored on a 
    separate thread that spawns additional child threads for each socket connection received.  The Server 
    class will need a function to be provided that is called on the new child thread when a new client 
    opens a connection.  All calls to send() and receive() are capable of throwing exceptions, i.e. if the
    network connection is closed unexpectedly, and error handling should consider this.  Examples of both 
    client and server use follow.
#########
    Example use of Server for a single-message service:
    
    def on_new_client(client: Protocol, address: Tuple[str,int], shutdown_event: threading.Event) -> None:    
        try:
            remote_ip_address = address[0]
            remote_port = address[1]
            try:
                remote_hostname_info = socket.gethostbyaddr(remote_ip_address)
                if remote_hostname_info and len(remote_hostname_info) > 1: remote_hostname = remote_hostname_info[0]
            except:
                remote_hostname = remote_ip_address

            # For a multi-message server, put a loop here and use timeout_in_seconds of 0, with a time.sleep()
            # for the 'None' case.  If the shutdown_event.is_set() triggers True, exit.  receive() will raise
            # an exception on any disconnected connection.

            msg = client.receive(require_response = False, timeout_in_seconds = 2)
            if msg is None:
                # Client has failed to send a command within timeout.
                return
            if msg.tag == "Do-Command":
                print("Command performed.")
                client.send("<Success/>")
            if msg.tag == "Disconnect":
                print("Client is disconnecting normally.")
                # client.disconnect() will be called automatically on return, and since we
                # are returning and not calling receive() anymore, there will be no exception
                # due to the remote disconnecting.
            else:
                client.send("<Error>An unrecognized command was received by the server.</Error>")
        except Exception as ex:
            logging.error(traceback.format_exc() + "\n\n" + "Error servicing client connection from " + str(remote_hostname) + ": " + str(ex))
        # client.disconnect() will be called automatically on return.

    # Establish a nicer logging format for long-running processes like a server.  Optionally, add %Y in front of %j to also
    # show the year.
    logging.basicConfig(format="%(asctime)s: %(message)s", datefmt="%j %H:%M:%S", level=logging.INFO)

    my_server = Server(PORT_NUMBER, on_new_client, name = "My Example XML Server", shutdown_timeout_in_seconds = 5)
    # loop forever or maybe just until ctrl-c.  In the exception handling or outside the loop...
    del my_server             # Causes all threads to shutdown gracefully.  Optional call, since also called after exit by GC.
#########
    Example use of a Client:

    try:
        with Client("MyServer.domain.com", PORT_NUMBER, send_disconnect = True) as connection:
            connection.send("<Request-Service />")
            response = connection.receive(require_response = True, timeout_in_seconds = 30)
            # response contains an ElementTree XML Element message to be handled now.
    except Exception as ex:
        logging.error(traceback.format_exc() + "\n\nError issuing request to server: " + str(ex))
"""
## Settings

USE_PRETTY_XML = True                    # Set True if debugging the connection for better XML readability.

## Dependencies

import os
import sys
import time
import socket
import select
import traceback
import threading
import logging
from typing import Callable, Tuple

import xml.etree.ElementTree as xml
import xml.dom.minidom as minidom
import xml.sax.saxutils as saxutils
Escape = saxutils.escape

## Helpers

logger = logging.getLogger()
info = logger.info

def to_pretty_xml(element_tree_root, omit_xml_declaration = True):
    """ Accepts an Element object and returns a pretty-formatted string. """

    # Return a pretty-printed XML string for the Element.    
    rough_string = xml.tostring(element_tree_root, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty1 = reparsed.toprettyxml(indent="    ")
    # Strip out multiple adjacent newlines:
    pretty2 = '\n'.join([line for line in pretty1.split('\n') if line.strip()])
    if omit_xml_declaration:
        start_idx = pretty2.find("<?xml")
        if start_idx >= 0:
            end_idx = pretty2.find("?>")
            if end_idx < 0: raise Exception("Found XML declaration to be omitted (<?xml ... ?>) but no ending text (?>) was found.")
            pretty2 = pretty2[:start_idx] + pretty2[(end_idx + 2):]
    return pretty2   

DISCONNECTED = -1
TIMEOUT = -2
def recv_with_timeout(on_socket, *args, timeout_in_seconds=0, **kwargs):    
    """ to be safe, also call on_socket.setblocking(0) once before this call. """

    ready = select.select([on_socket], [], [], timeout_in_seconds)    
    if ready[0]:
        ret = on_socket.recv(*args, **kwargs)
        # In the event that we've been disconnected, recv() returns None.
        # It seems also possible that a ConnectionError will be raised.
        if ret is None or len(ret) == 0: return DISCONNECTED
        return ret
    return TIMEOUT

## Protocol class

class Protocol:
    """ The Protocol class implements the lowest custom layer in the XML Socket stack and is responsible
        for packaging individual messages with a prefix.  It also monitors connection state and is
        responsible for the sending of <Disconnect /> messages upon intentional closing.
    """

    connection = None
        
    def __init__(self, attach_socket: socket.socket, send_disconnect: bool = True):
        """ Use either the Server class or Client() function to create a new Protocol object instead 
            of creating one directly.       
        
            If send_disconnect is True, then a message "<Disconnect />" will be transmitted before 
            the intentional closing of the connection such as exiting a 'with' clause or del
            on the Protocol object.
        """

        self._rx_stage = 0
        self._rx_msg_len = 0
        self._rx_packet = None
        self.send_disconnect = send_disconnect
        self.connection = attach_socket        

    def __del__(self):
        if self.connection is not None:
            del self.connection 
            self.connection = None

    def __enter__(self):
        return self
    
    @property
    def is_disconnected(self): 
        return self.connection is None or self._rx_stage == DISCONNECTED

    @property
    def is_connected(self):
        return self.connection is not None and self._rx_stage != DISCONNECTED

    def send(self, msg):
        """ Transmit a message, which can be either a string or an ElementTree Element.  If a disconnection has
            been detected, then send() raises a ConnectionError.  Caller can also check the is_disconnected or 
            is_connected property before or after the call.
        """

        if isinstance(msg, xml.Element):
            msg = to_pretty_xml(msg, omit_xml_declaration = True) if USE_PRETTY_XML else xml.tostring(msg, 'utf-8')            
        if not(isinstance(msg, str)):
            raise TypeError("xml_socket.Protocol.Send() requires either an xml Element (from xml.etree.ElementTree) or a string message.  A " + str(type(msg)) + " was provided instead.")
        if self._rx_stage == DISCONNECTED: raise ConnectionError("Connection has already been closed.")        
        if self.connection is None:
            raise Exception("xml_socket.Protocol.Send() failed because no connection was established.")        
        encoded = msg.encode("utf-8")
        msg_len = len(encoded)
        try:
            self.connection.sendall(msg_len.to_bytes(4, 'big'))
            self.connection.sendall(encoded)
            # info("Message sent.")
        except:
            self._rx_stage = DISCONNECTED
            raise
            
    def receive(self, require_response = False, timeout_in_seconds = None):        
        """ Call receive() to check for a message from the other host.  If timeout_in_seconds is specified (and not 
            None), then receive() will only wait for the given number of seconds for a response.  If timeout_in_seconds
            is 0, then receive() checks for waiting data and if a message is not received it returns immediately.
            Additional calls to receive() will continue assembling the same message.  
            
            If require_response is True then an exception is thrown in the event that the timeout expires without a 
            completed message.  If a message is received, it is decoded and parsed, and an ElementTree Element is 
            returned.
            
            If require_response is False, and a timeout occurs then None is returned.

            In all cases, if an unexpected disconnection is detected, then an exception is thrown.  To initiate a
            normal disconnect, del the Protocol or exit the holding "with" clause.  To avoid exceptions for normal
            disconnects, the caller may also want to respond to a <Disconnect/> message by initiating a normal 
            disconnect (del or with clause).
        """

        if self._rx_stage == DISCONNECTED: raise ConnectionError("Connection has already been closed.")
        if self.connection is None:
            raise Exception("xml_socket.Protocol.receive() failed because no connection was established.")        
        if self._rx_stage == 0:
            self._rx_packet = b''
            self._rx_stage = 1
            self._rx_msg_len = 4
        
        # This part is the same whether in rx_stage 1 or 2, receive however many bytes we know to expect next.
        start_time = time.time()
        elapsed = 0
        while len(self._rx_packet) < self._rx_msg_len:
            remaining_len = self._rx_msg_len - len(self._rx_packet)
            if timeout_in_seconds is not None:
                chunk = recv_with_timeout(self.connection, remaining_len, 
                    timeout_in_seconds=timeout_in_seconds - elapsed if elapsed < timeout_in_seconds else 0)
                elapsed = time.time() - start_time
            else:
                chunk = self.connection.recv(remaining_len)
                if chunk is None or len(chunk) == 0: chunk = DISCONNECTED
            if chunk == DISCONNECTED:
                self._rx_stage = DISCONNECTED
                raise ConnectionError("Connection has been closed.")
            if chunk == TIMEOUT:
                if require_response: raise Exception("Timeout without receiving response.")
                return None
            self._rx_packet += chunk
        if len(self._rx_packet) != self._rx_msg_len: 
            raise Exception("xml_socket.Protocol.Receive() failed: expected to receive " + str(self._rx_msg_len) 
                + " bytes from socket, but received " + len(self._rx_packet) + " bytes instead.")

        # Now process the received bytes according to our state.

        if self._rx_stage == 1:
            self._rx_msg_len = int.from_bytes(self._rx_packet, "big", signed=False)
            self._rx_stage = 2
            self._rx_packet = b''
            elapsed = time.time() - start_time
            return self.receive(require_response=require_response, timeout_in_seconds=timeout_in_seconds - elapsed)       
        
        if self._rx_stage == 2:
            as_str = self._rx_packet.decode("utf-8")
            self._rx_stage = 0
            # info("Message received: " + str(as_str))
            return xml.fromstring(as_str)

    def disconnect(self):
        if self.is_connected:
            if self.send_disconnect: 
                # If the socket has already disconnected, then this call will result in an exception.  Let's ignore it...
                try:
                    self.send("<Disconnect />")        
                except:
                    pass
            self._rx_stage = DISCONNECTED
            del self.connection
            self.connection = None

    def __exit__(self, otype, value, traceback):
        self.disconnect()

    def __del__(self):
        self.disconnect()

class Server:
    """ Server provides the server-side of an xml_socket connection.  The Server class is created and monitors
        for incoming connections on the specified port on an independent thread that Server creates.  For each
        incoming connection request, Server will launch a new thread and call the provided on_new_client callback
        function to handle the connection.
    """

    def __init__(self, 
        port: int, 
        on_new_client: Callable[[Protocol, Tuple[str,int], threading.Event], None],
        name: str = "XML Server", 
        shutdown_timeout_in_seconds: int = None
        ):
        """ on_new_client must provide a callback function for whenever a new connection is opened by a client.  It
            will be called from the Server's separate thread.

            on_new_client should accept arguments as: Protocol object, remote address, and shutdown event.  
            The shutdown event provides a threading.Event() object that is to be monitored and should cause an immediate 
            shutdown when .is_set() returns true.  The shutdown event is set by the closure of the Server object (by
            del or closing a "with" clause).

            The port should specify the server port number on which to listen for incoming connections.

            The optional 'name' argument is provided to clarify which Server is logging.  The default is "XML Server".
            The name argument is also applied to threads created by the Server, with workers having " worker #X"
            appended.

            The optional shutdown_timeout_in_seconds argument provides a timeout on closure to wait for the join() of
            threads, generally which are processing the on_new_client callback.  Since threads created by
            Server are marked as Daemon threads, they will be forcefully shutdown after shutdown_timeout_in_seconds
            expires.  The default shutdown_timeout_in_seconds will wait indefinitely.            
        """

        self.name = name
        self.shutdown_timeout_in_seconds = shutdown_timeout_in_seconds
        self.on_new_client = on_new_client
        
        self.shutdown_event = threading.Event()
        self.shutdown_event.clear()
        
        self._main_thread = threading.Thread(target=self.run_main_thread, name=name, args=[port])
        self._main_thread.setDaemon(True)          # We will allow N seconds for graceful shutdown, then exit.
        self._main_thread.start()

    def run_main_thread(self, port):
        logging.info(f"Starting {self.name}...")               

        service_threads = []
        thread_id = 1
            
        # Create a server socket for incoming requests
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Bind the socket to a public host on the incoming port            
            server_socket.bind(('', port))
            # become a server socket
            server_socket.listen(5)
            
            logging.info(f"{self.name} has started on port " + str(port) + ".")
            while (not(self.shutdown_event.is_set())):
                ## Check for incoming service requests
                readable, writable, errored = select.select([server_socket], [], [], 0)

                for read_socket in readable:
                    if read_socket == server_socket:                                                
                        ## Accept incoming service requests and delegate them to an available node.        
                        (client_socket, address) = server_socket.accept()                        
                        client_protocol = Protocol(attach_socket=client_socket)                        

                        def new_client_launcher(handler, client, address, shutdown_event):
                            try:
                                handler(client, address, shutdown_event)
                            except:
                                try:
                                    client.disconnect()
                                except:
                                    pass
                                raise
                            client.disconnect()

                        new_thread = threading.Thread(target=new_client_launcher, 
                            name=self.name + " worker #" + str(thread_id),
                            args=[self.on_new_client, client_protocol, address, self.shutdown_event])
                        thread_id += 1
                        new_thread.setDaemon(True)          # We will allow N seconds for graceful shutdown, then exit.
                        new_thread.start()
                        service_threads.append(new_thread)

                time.sleep(0.010)
                
            logging.info(f"{self.name} is shutting down.")
            start = time.time()
            for thread in service_threads:
                remaining = time.time() - start
                if remaining <= 0: break
                thread.join(timeout=self.shutdown_timeout_in_seconds)
                
            logging.info(f"{self.name} main thread has shut down.")
        except Exception as ex:
            logging.error(traceback.format_exc() + "\n\n" + str(self.name) + " error: " + str(ex))

    def close(self):    
        if self.shutdown_event is not None:
            self.shutdown_event.set()
            # The main thread will call join() on all new threads that it has spawned, and so joining on 
            # the main thread here will also ensure all child threads were closed.
            self._main_thread.join(self.shutdown_timeout_in_seconds)            
            self.shutdown_event = None
            self._main_thread = None

    def __enter__(self):
        return self

    def __del__(self):
        self.close()

    def __exit__(self, otype, value, traceback):
        self.close()

def Client(remote_host_name, remote_port, send_disconnect = True):
    """ Client() returns a Protocol object for use as the client-side of a connection
        to an xml_socket server.  If send_disconnect is True, then a message "<Disconnect />" 
        will be transmitted before the intentional closing of the connection.
    """
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.connect((remote_host_name, remote_port))
    return Protocol(connection, send_disconnect= send_disconnect)
