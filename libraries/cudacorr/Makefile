CC = nvcc
CFLAGS = -Xcompiler -fopenmp -Xcompiler -fPIC -lm -O3
LDFLAGS = 
LOCAL_INC_DIRS = ./include
LOCAL_LIB_DIRS = ./lib 
EXT_INC_DIRS = $(PREFIX)/include
EXT_LIB_DIRS = $(PREFIX)/lib
SOURCES = ./src/*.cu

libcudacorr: 
	$(CC) -shared $(CFLAGS) $(LDFLAGS) -I$(LOCAL_INC_DIRS) -I$(EXT_INC_DIRS) -L$(EXT_LIB_DIRS) -L$(LOCAL_LIB_DIRS) $(SOURCES) -o lib/libcudacorr.so
	mkdir -p $(PREFIX)/include/atsec/
	mkdir -p $(PREFIX)/lib/
	cp lib/libcudacorr.so $(PREFIX)/lib/
	cp include/*.h $(PREFIX)/include/atsec/
