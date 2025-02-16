#include "mem/cache/prefetch/rl_naive.hh"

#include "base/logging.hh"
#include "debug/HWPrefetch.hh"
#include "params/RLNaivePrefetcher.hh"

#include <boost/algorithm/string.hpp>
#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <sstream>

RLNaivePrefetcher::RLNaivePrefetcher(const RLNaivePrefetcherParams *p) 
    : QueuedPrefetcher(p)
{
	std::cout << "Prefetching!";
}


size_t write_data(void *contents, size_t size, size_t nmemb, void *userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}


void
RLNaivePrefetcher::calculatePrefetch(const PrefetchInfo &pfi,
		std::vector<AddrPriority> &addresses)
{
	if(!pfi.hasPC()){
		DPRINTF(HWPrefetch, "Ignoring request with no PC\n");
		return;
	}
	// std::cout << "Prefetching!\n";

	CURL *curl;
	CURLcode res;
    std::string readBuffer;
    std::ostringstream addrStr;
    addrStr << pfi.getAddr();
    std::ostringstream pcStr;
    pcStr << pfi.getPC();
    
    // print to file 
    // ios::out specifies output file; ios::app specifies append
    // std::ofstream output_file;
    // output_file.open ("/gem5/address_pc_list.txt", std::ios::out | std::ios::app); 
    // output_file << addrStr.str() << "," << pcStr.str() << "\n";
    // output_file.close();


    /* In windows, this will init the winsock stuff */
    curl_global_init(CURL_GLOBAL_ALL);

    /* get a curl handle */
    curl = curl_easy_init();
    if(curl) {
        /* First set the URL that is about to receive our POST. This URL can
       just as well be a https:// URL if that is what should receive the
       data. */
        curl_easy_setopt(curl, CURLOPT_URL, "localhost:8080");
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_POST, 1);

        std::string strData;
        strData = "address=" + addrStr.str();
        strData += "&pc=" + pcStr.str();
        /* Now specify the POST data */
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, strData.c_str());

        /* Perform the request, res will get the return code */
        res = curl_easy_perform(curl);
        /* Check for errors */
        if(res != CURLE_OK) {
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
        }

        // the server returns a string of addresses split by ','
        // we need to split the string and convert the addresses to the Addr type
        std::vector<std::string> splitAddrStr;
        boost::split(splitAddrStr, readBuffer, boost::is_any_of(","));
	if (readBuffer.length() > 0) {
	    for (int i = 0; i < splitAddrStr.size(); i++) {
	        Addr returnedAddress = std::stoul(splitAddrStr[i]);
	        addresses.push_back(AddrPriority(returnedAddress, 0));
	    }
	}

        // confirm that it worked (we can get rid of this)
        // std::cout << "* size of the vector: " << splitAddrStr.size() << "\n";
        // std::cout << splitAddrStr[splitAddrStr.size() - 1] << "\n";


        /* always cleanup */
        curl_easy_cleanup(curl);
    }
    curl_global_cleanup();

}

RLNaivePrefetcher*
RLNaivePrefetcherParams::create()
{
	return new RLNaivePrefetcher(this);
}


