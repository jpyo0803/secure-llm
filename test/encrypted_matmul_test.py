import torch

K = 2048

for i in range(100000):
  x = torch.randint(-128, 128, (1, K), dtype=torch.int32)
  # x[0, 0] = 127
  y = torch.randint(-128, 128, (K, 1), dtype=torch.int32)

  a = torch.randint(-(2**31), 2**31, (1, K), dtype=torch.int32)
  # a[0, 0] = 2**31 - 1


  enc_x = x + a
  print("x[0, 0] = ", x[0, 0])
  print("a[0, 0] = ", a[0, 0])
  print("enc_x[0, 0] = ", enc_x[0, 0])

  unblind_factor = torch.matmul(a, y)
  enc_z = torch.matmul(enc_x, y)
  z = enc_z - unblind_factor

  z_orig = torch.matmul(x, y)
  print(z_orig)
  print(z)
  assert torch.equal(z_orig, z)
  print(f"{i}-th Passed!")