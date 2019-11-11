#include "mem/cache/prefetch/rl_naive.hh"

#include "base/logging.hh"
#include "debug/HWPrefetch.hh"
#include "params/RLNaivePrefetcher.hh"

#include <iostream>
#include <curl/curl.h>

RLNaivePrefetcher::RLNaivePrefetcher(const RLNaivePrefetcherParams *p) 
    : QueuedPrefetcher(p)
{
	std::cout << "Prefetching!";    
}

void
RLNaivePrefetcher::calculatePrefetch(const PrefetchInfo &pfi,
		std::vector<AddrPriority> &addresses)
{
	if(!pfi.hasPC()){
		DPRINTF(HWPrefetch, "Ignoring request with no PC\n");
		return;
	}
	std::cout << "Prefetching!\n";

	CURL *curl;
	CURLcode res;


     /* In windows, this will init the winsock stuff */
    curl_global_init(CURL_GLOBAL_ALL);

    /* get a curl handle */
    curl = curl_easy_init();
    if(curl) {
        /* First set the URL that is about to receive our POST. This URL can
       just as well be a https:// URL if that is what should receive the
       data. */
        curl_easy_setopt(curl, CURLOPT_URL, "localhost:8080");
        /* Now specify the POST data */
//        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "name=daniel&project=curl");

        /* Perform the request, res will get the return code */
        res = curl_easy_perform(curl);
        /* Check for errors */
        if(res != CURLE_OK)
            fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));

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


