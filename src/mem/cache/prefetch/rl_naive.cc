#include "mem/cache/prefetch/rl_naive.hh"

#include "base/logging.hh"
#include "debug/HWPrefetch.hh"
#include "params/RLNaivePrefetcher.hh"

#include <iostream>

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
	std::cout << "Prefetching!";
}

RLNaivePrefetcher*
RLNaivePrefetcherParams::create()
{
	return new RLNaivePrefetcher(this);
}

