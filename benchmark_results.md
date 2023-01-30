Running `test_and_benchm.py` on different architectures

RedLionfish v0.5

Values provided are in seconds. Lower is better.

| Machine                       | CPU                              | GPU                 | 256x256x256 | 256x256x256 | 1024x1024x1024 | 1024x1024x1024  |
|-------------------------------|----------------------------------|---------------------|-------------|-------------|----------------|-----------------|
|                               |                                  |                     |     CPU     |     GPU     |      CPU       |      GPU        |
| Laptop HP Spectre x360 Pro G2 | i5-6300                          | Intel HD            |      19     |      34     |      116       |      292 b      |
| Laptop Dell Precision 3561    | Intel Core i7 11850H             | NVIDIA T600         |      10     |     4.6     |       47       |      30 b       |
| Desktop PC                    | AMD FX 6300 6 core               | NVIDIA GTX 1060 3Gb |      31     |     9.7     |      186       |      38.7 b     |
| Guacamole VM - ELS            | Intel Xeon (Skylake IBRS)        | NVIDIA Quadro P4000 |      8      |     3.5     |      38.6      |       9.5 b     |
| Baskerville Tier 2            | Intel Xeon Platinum 8360Y 2.4GHz | A100-SXH4-40Gb      |     5.9     |     3.3     |      26.2      |       5.6 b     |
| Laptop Asus Vivobook X3400PH  | Intel Core i5-11300H 3.10GHz     | NVIDIA GeForce GTX 1650 | 16.2    |     4.02    |      155       |      33.8 b     |

b: blocking algorithm was used