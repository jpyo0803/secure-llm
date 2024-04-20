#include "aes_stream.h"
#include <iostream>
#include <vector>

using namespace std;

unsigned char init_seed[AES_STREAM_SEEDBYTES] = {0x00};
aes_stream_state producer_PRG;

int main() {  
  aes_stream_init(&producer_PRG, init_seed);

  size_t buf_len = 128;
  vector<unsigned char> buf(buf_len);

  aes_stream(&producer_PRG, buf.data(), buf_len);

  for (int i = 0; i < buf_len; ++i) {
    cout << i << "-th random byte : " << (int)buf[i] << endl;
  }

  return 0;
}