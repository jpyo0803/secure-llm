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

  if (buf_len < 128) {
    for (int i = 0; i < buf_len; ++i) {
      cout << i << "-th random byte : " << (int)buf[i] << endl;
    }
  }

  cout << "Generating " << buf_len << " PRNG, Latency = " << chrono::duration_cast<chrono::microseconds>(end - begin).count() << "[µs]" << endl;


  return 0;
}