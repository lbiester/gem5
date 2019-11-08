#ifndef __MEM_CACHE_PREFETCH_LSTMNAIVE_HH__
#define __MEM_CACHE_PREFETCH_LSTMNaive_HH__

#include <vector>

#include "base/types.hh"
#include "mem/cache/prefetch/queued.hh" 
#include "mem/packet.hh"

struct LSTMNaivePrefetcherParams;

class LSTMNaivePrefetcher : public QueuedPrefetcher
{
	public: 
		LSTMNaivePrefetcher(const LSTMNaivePrefetcherParams *p);		
		
		void calculatePrefetch(const PrefetchInfo &pfi,
					std::vector<AddrPriority> &addresses) override; 
};

#endif //__MEM_CACHE_PREFETCH_QUEUED_HH__

