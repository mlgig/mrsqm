**Q**: gcc failed with this error: 
```
gcc -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/opt/anaconda3/envs/comp47350py38/include -arch x86_64 -I/opt/anaconda3/envs/comp47350py38/include -arch x86_64 -I/opt/anaconda3/envs/comp47350py38/include/python3.8 -c mrsqm_wrapper.cpp -o build/temp.macosx-10.9-x86_64-3.8/mrsqm_wrapper.o -Wall -Ofast -g -std=c++11 -mfpmath=both -ffast-math
error: unknown FP unit 'both'
error: command 'gcc' failed with exit status 1
```

**A**: Go to file mrsqm/mrsqm/setup.py and in line " extra_compile_args=["-Wall", "-Ofast", "-g", "-std=c++11", , "-mfpmath=both", "-ffast-math"],"
remove the option , "-mfpmath=both".
