# sdpaFwd Split-Architecture Smoke Matrix

| group | name | dim | heads | seq | compile | compile_ms | error |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| baseline | 7B | 4096 | 32 | 512 | ok | 240.4 |  |
| baseline | 10B | 4096 | 32 | 512 | ok | 225.8 |  |
| baseline | 20B | 5120 | 40 | 512 | ok | 252.3 |  |
| baseline | 30B | 6144 | 48 | 512 | ok | 255.7 |  |
| baseline | 40B | 5120 | 40 | 512 | ok | 255.6 |  |
| larger | 50B | 6144 | 48 | 512 | ok | 258.0 |  |
| larger | 60B | 6144 | 48 | 512 | ok | 258.9 |  |
