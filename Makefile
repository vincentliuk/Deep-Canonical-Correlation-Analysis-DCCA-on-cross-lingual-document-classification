OPTFLAGS = -O3 -march=nocona -mfpmath=sse -msse3 -Wuninitialized -flto
CFLAGS = -std=c++0x -I/usr/nikola/pkgs/intel/.2011.2.137/mkl/include -I/g/ssli/software/pkgs/boost_1_49_0 -c -DNDEBUG -D__LINUX $(OPTFLAGS) 
CPP = g++

MKLLINKFLAGS = -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread

LINKOPTFLAGS = -O3 -flto=4 -fwhole-program
LINKFLAGS = -static $(LINKOPTFLAGS) $(MKLLINKFLAGS)

ODIR=obj

_OBJ = Matrix.o main.o TerminationCriterion.o LBFGS.o
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

dcca.exe : $(OBJ)
	$(CPP) -o $@ $^ $(LINKFLAGS)

obj/Matrix.o : Matrix.h Random.h Matrix.cpp
	$(CPP) $(CFLAGS) -o $@ Matrix.cpp

obj/main.o : $(wildcard *.cpp *.h)
	$(CPP) $(CFLAGS) -o $@ main.cpp

obj/TerminationCriterion.o : $(wildcard *.cpp *.h)
	$(CPP) $(CFLAGS) -o $@ TerminationCriterion.cpp

obj/LBFGS.o : $(wildcard *.cpp *.h)
	$(CPP) $(CFLAGS) -o $@ LBFGS.cpp

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~
