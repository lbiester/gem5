import http.server
import socketserver
from urllib.parse import parse_qs

value = 0
addresses_sent = []

class TorchHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        # get data from request
        # if we need more than super basic data (like address), we can consider
        # using JSON
        length = int(self.headers.get("content-length"))
        field_data = self.rfile.read(length)
        field_dict = parse_qs(field_data.decode('ascii'))
        address = int(field_dict["address"][0])
        print("Address sent:", address)
        
        global addresses_sent
        addresses_sent.append(str(address))


        # sketchy use of global in python but I can't think of another simple
        # way to do this
        # the network itself could be stored in a global variable
        global value
        value += 1

        # this library is extrordinarily annoying and won't write back a number
        # without this mess...
        
        self.wfile.write(b"\n")
        self.wfile.write(bytes(",".join(addresses_sent), "utf-8"))

PORT = 8080
Handler = TorchHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
