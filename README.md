# pyLibrary
Some Python utilities.

- gpu_util.py: convenience functions for writing cupy and cupyx code that is easily interchangable with cpu-based code or cpu-only situations.
- xml_socket.py: provides simple sockets-based TCP/IP server & client classes that communicate using an XML protocol.  Interpretation of the messages is up to the caller.
- math/
  - lsq2dpoly.py: least-squares 2D fitting with arbitrary order polynomial, supporting a mask function for the use of omitted/included points on an otherwise regular grid.
  - special_functions.py: mathematical functions such as tri(), sinc(), and gaus() in 1 and 2-D.
  - wbFFT.py: provides 1 and 2-D FFT and IFFT where both the axis and the data are specified.  This can be handy for plotting where you don't want to think about the data's axes or the frequency axes and instead just define them once. *
  
 * this one needs some test cases and a little TLC, may not work properly in all use cases.
