#include "aes_stream.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace std;

unsigned char init_seed[AES_STREAM_SEEDBYTES] = {0x00};
aes_stream_state producer_PRG;

int main() {  
  aes_stream_init(&producer_PRG, init_seed);

  size_t buf_len = 8192*8192;
  vector<unsigned char> buf(buf_len);

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();
  aes_stream(&producer_PRG, buf.data(), buf_len);
  chrono::steady_clock::time_point end = chrono::steady_clock::now();

  cout << "Generating " << buf_len << " PRNG, Latency = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;

  size_t buf_len2 = buf_len;
  vector<unsigned char> buf2(buf_len2);

  begin = chrono::steady_clock::now();
  GetCPRNG(buf2.data(), buf_len2);
  end = chrono::steady_clock::now();

  cout << "Generating " << buf_len2 << " PRNG, Latency = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;

  return 0;
}