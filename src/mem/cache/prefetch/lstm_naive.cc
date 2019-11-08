#include "mem/cache/prefetch/lstm_naive.hh"

#include "base/logging.hh"
#include "debug/HWPrefetch.hh"
#include "params/LSTMNaivePrefetcher.hh"

#include <iostream>

LSTMNaivePrefetcher::LSTMNaivePrefetcher(const LSTMNaivePrefetcherParams *p) 
    : QueuedPrefetcher(p)
{
	std::cout << "Prefetching!";    
}

void
LSTMNaivePrefetcher::calculatePrefetch(const PrefetchInfo &pfi,
		std::vector<AddrPriority> &addresses)
{
	if(!pfi.hasPC()){
		DPRINTF(HWPrefetch, "Ignoring request with no PC\n");
		return;
	}
	std::cout << "Prefetching!";
}

LSTMNaivePrefetcher*
LSTMNaivePrefetcherParams::create()
{
	return new LSTMNaivePrefetcher(this);
}

