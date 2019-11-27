import http.server
import socketserver
import sys
from io import BytesIO
from urllib.parse import parse_qs

value = 0
addresses_sent = []
pcs_sent = []

OUTFILE_ADDR = "spec_stats/address_pc_experiment/address.out"
OUTFILE_PC = "spec_stats/address_pc_experiment/pc.out"

class TorchHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        # this is to avoid writing to a file each time... we can make a get
        # request to write back all of the PCs/Addresses sent to a file
        if len(sys.argv) > 1:
            global addresses_sent
            global pcs_sent
            with open(OUTFILE_ADDR + "." + sys.argv[1], "w") as f:
                for address in addresses_sent:
                    f.write(str(address) + "\n")
            with open(OUTFILE_PC + "." + sys.argv[1], "w") as f:
                for pc in pcs_sent:
                    f.write(str(pc) + "\n")

    def do_POST(self):
        # global variables
        global value
        global addresses_sent
        global pcs_sent

        self.send_response(200)
        self.end_headers()
        # get data from request
        # if we need more than super basic data (like address), we can consider
        # using JSON
        length = int(self.headers.get("content-length"))
        field_data = self.rfile.read(length)
        field_dict = parse_qs(field_data.decode('ascii'))

        if "reset_state" in field_dict:
            # special case: the config file will initially call this with
            # rest_state to reset the server state between runs
            value = 0
            addresses_sent = []
            pcs_sent = []
            print("resetting server state")
            return
        address = int(field_dict["address"][0])
        pc = int(field_dict["pc"][0])

        addresses_sent.append(str(address))
        pcs_sent.append(str(pc))


        # sketchy use of global in python but I can't think of another simple
        # way to do this
        # the network itself could be stored in a global variable
        value += 1

        # this library is extrordinarily annoying and won't write back a number
        # without this mess...

        addresses_test = ["1234", "5678"]
        response = BytesIO()
        response.write(bytes(",".join(addresses_test), "utf-8"))
        self.wfile.write(response.getvalue())


PORT = 8080
Handler = TorchHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
