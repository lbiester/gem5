#ifndef __MEM_CACHE_PREFETCH_RLNAIVE_HH__
#define __MEM_CACHE_PREFETCH_RLNAIVE_HH__

#include <vector>

#include "base/types.hh"
#include "mem/cache/prefetch/queued.hh" 
#include "mem/packet.hh"

struct RLNaivePrefetcherParams;

class RLNaivePrefetcher : public QueuedPrefetcher
{
	public: 
		RLNaivePrefetcher(const RLNaivePrefetcherParams *p);		
		
		void calculatePrefetch(const PrefetchInfo &pfi,
					std::vector<AddrPriority> &addresses) override;

        size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp);
};

#endif //__MEM_CACHE_PREFETCH_RLNAIVE_HH__


