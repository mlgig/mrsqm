**Q**: gcc failed with this error: 
```
gcc -Wno-unused-result -Wsign-compare -Wunreachable-code -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -I/opt/anaconda3/envs/comp47350py38/include -arch x86_64 -I/opt/anaconda3/envs/comp47350py38/include -arch x86_64 -I/opt/anaconda3/envs/comp47350py38/include/python3.8 -c mrsqm_wrapper.cpp -o build/temp.macosx-10.9-x86_64-3.8/mrsqm_wrapper.o -Wall -Ofast -g -std=c++11 -mfpmath=both -ffast-math
error: unknown FP unit 'both'
error: command 'gcc' failed with exit status 1
```

**A**: Go to file mrsqm/mrsqm/setup.py and in line " extra_compile_args=["-Wall", "-Ofast", "-g", "-std=c++11", , "-mfpmath=both", "-ffast-math"],"
remove the option , "-mfpmath=both".


**Q**: fftw3 error during compilation: 
```
//usr/local/lib/libfftw3.a(assert.o): relocation R_X86_64_PC32 against symbol `stdout@@GLIBC_2.2.5' can not be used when making a shared object; recompile with -fPIC
    /home/thachln/anaconda3/envs/vistamilk/compiler_compat/ld: final link failed: bad value
    collect2: error: ld returned 1 exit status
    error: command 'g++' failed with exit status 1
```


**A**: Reinstall fftw3 following this guide http://micro.stanford.edu/wiki/Install_FFTW3. 
