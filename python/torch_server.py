import http.server
import socketserver
from io import BytesIO
from urllib.parse import parse_qs

from python import util
from python.qlearning import QLearningPrefetcher
from python.table_bandits import ContextBandit

prefetcher = None

class TorchHandler(http.server.BaseHTTPRequestHandler):

    def do_POST(self):
        # global variables
        global prefetcher

        self.send_response(200)
        self.end_headers()
        # get data from request
        # if we need more than super basic data (like address), we can consider
        # using JSON
        length = int(self.headers.get("content-length"))
        field_data = self.rfile.read(length)
        field_dict = parse_qs(field_data.decode('ascii'))

        if "rl_prefetcher" in field_dict:
            # special case: the config file will initially call this with
            # rl_prefetcher to reset the server state between runs and initialize the correct prefetcher to be used
            print("Initializing prefetcher")
            spec_program = field_dict["spec_program"][0]

            state_vocab, action_vocab = util.load_vocab(spec_program)
            if field_dict["rl_prefetcher"][0] == "table_bandits":
                prefetcher = ContextBandit(state_vocab, action_vocab)
            elif field_dict["rl_prefetcher"][0] == "table_q":
                prefetcher = QLearningPrefetcher(state_vocab, action_vocab)
            else:
                raise Exception("Unsupported prefetcher")
            print("Done initializing prefetcher")
            return
        assert(prefetcher is not None)
        address = int(field_dict["address"][0])
        pc = int(field_dict["pc"][0])

        prefetch_address = prefetcher.select_action((address, pc))

        addresses_test = [str(prefetch_address)]
        response = BytesIO()
        response.write(bytes(",".join(addresses_test), "utf-8"))
        self.wfile.write(response.getvalue())


PORT = 8080
Handler = TorchHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()


