cd secure_llm && \
make -f Makefile.no_sgx clean && \
make -f Makefile.no_sgx -j && \
cp secure_llm_no_sgx.so .. && \
cd .. && \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so python3 smoothquant_generation.py
